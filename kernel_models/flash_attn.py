"""
kernel_models/flash_attn.py — Python model of kernels/flash_attn.cu

Models the exact tile geometry, loop structure, and online softmax of the
CUDA kernel using NumPy.  Each function corresponds 1-to-1 to a CUDA
function; comments call out the matching CUDA line where the mapping is
non-obvious.

Tile constants (matching flash_attn.cu):
    Br          = 16   kFlashBr      — Q row tile (one m16 MMA row tile)
    Bc_col      =  8                 — KV sequence column tile (one n8 MMA col tile)
    k_tile_size = 16                 — depth tile for QKᵀ MMA (m16n8k16)
    v_tile_cols =  8                 — D-column tile for PV MMA (m16n8k8)

Current loop order in warp_flash_attn (outer → inner):
    rb : 0 .. q_rows/16     Q row tiles     (CUDA: warp_id per block)
    cb : 0 .. kv_cols/8     KV column tiles (CUDA line ~115)
    k  : 0 .. D/16          depth tiles     (CUDA line ~122) — all accumulated before softmax
    co : 0 .. D/8           V D-column tiles (CUDA line ~265)

Online softmax fires once per (rb, cb) after the full QKᵀ[rb,cb] dot product
has been accumulated across all k-tiles.  This matches correct Flash Attention.

Register allocation (v2):
    mi_reg [q_rows]    — running per-row max, held in registers across all KV tiles
    li_reg [q_rows]    — running per-row sum-of-exp, held in registers
    o_reg  [q_rows, D] — output accumulator, held in registers; single HBM write at end
"""

import numpy as np

# ── Tile constants ────────────────────────────────────────────────────────────
Br          = 16   # kFlashBr
Bc_col      =  8   # KV sequence tile width (n8 MMA)
k_tile_size = 16   # depth tile for QKᵀ (m16n8k16)
v_tile_cols =  8   # D-column tile for PV (m16n8k8)
kFlashWarps =  1


# ── smem_load_frag_a  (16×16 Q tile, ldmatrix.x4) ────────────────────────────
def smem_load_frag_a(Q_smem: np.ndarray, rb: int, k: int) -> np.ndarray:
    """
    [Br, k_tile_size] slice of Q_smem for tile (rb, k).
    CUDA: ldmatrix.sync.aligned.m8n8.x4 — 4 registers/lane, 16×16 fp16 A-fragment.
    """
    row  = rb * Br
    col  = k  * k_tile_size
    tile = Q_smem[row:row + Br, col:col + k_tile_size]
    out  = np.zeros((Br, k_tile_size), dtype=np.float32)
    out[:tile.shape[0], :tile.shape[1]] = tile
    return out


# ── smem_load_frag_b  (8×16 K tile, ldmatrix.x2) ─────────────────────────────
def smem_load_frag_b(K_smem: np.ndarray, cb: int, k: int) -> np.ndarray:
    """
    [Bc_col, k_tile_size] slice of K_smem for tile (cb, k).
    CUDA: ldmatrix.sync.aligned.m8n8.x2 — 2 registers/lane, 16×8 fp16 B-fragment.
    K layout in smem: [kv_cols, D].
    """
    row  = cb * Bc_col
    col  = k  * k_tile_size
    tile = K_smem[row:row + Bc_col, col:col + k_tile_size]
    out  = np.zeros((Bc_col, k_tile_size), dtype=np.float32)
    out[:tile.shape[0], :tile.shape[1]] = tile
    return out


# ── smem_load_frag_b8_trans  (8×8 V tile, ldmatrix.x1.trans) ─────────────────
def smem_load_frag_b8_trans(V_smem: np.ndarray, cb: int, co: int) -> np.ndarray:
    """
    [Bc_col, v_tile_cols] slice of V_smem for tile (cb, co).
    CUDA: ldmatrix.x1.trans — 1 register/lane.  The .trans instruction reorders
    register lanes for the m16n8k8 MMA B-operand layout; at the matrix level the
    data read is still V[cb*8:+8, co*8:+8] without any mathematical transpose.
    v_tile = V + cb*8*DHeadDim + co*8  (CUDA line ~268)
    """
    row  = cb * Bc_col
    col  = co * v_tile_cols
    tile = V_smem[row:row + Bc_col, col:col + v_tile_cols]
    out  = np.zeros((Bc_col, v_tile_cols), dtype=np.float32)
    out[:tile.shape[0], :tile.shape[1]] = tile
    return out   # [Bc_col, v_tile_cols] — no matrix transpose


# ── warp_flash_attn ───────────────────────────────────────────────────────────
def warp_flash_attn(
    Q_smem: np.ndarray,    # [active_q_rows,  D] fp32
    K_smem: np.ndarray,    # [active_kv_cols, D] fp32
    V_smem: np.ndarray,    # [active_kv_cols, D] fp32
    mi_reg: np.ndarray,    # [active_q_rows]     fp32  running row max    (registers)
    li_reg: np.ndarray,    # [active_q_rows]     fp32  running row sumexp (registers)
    o_reg:  np.ndarray,    # [active_q_rows, D]  fp32  running output     (registers)
) -> None:
    """
    Direct model of warp_flash_attn() in flash_attn.cu.  Mutates mi_reg/li_reg/o_reg.

    mi_reg, li_reg, o_reg are register-held accumulators initialized by the caller
    (flash_attn_kernel) before the kv_tile loop and written to HBM once afterward.
    This is the v2 change: no HBM round-trip for O/Mi/Li per KV tile.

    Loop structure (exact match to CUDA):
        for rb (Q row tiles, one per warp in CUDA)     — CUDA: warp_id
          for cb (KV sequence col tiles)               — CUDA line ~115
            c_frag = 0  ← reset once per (rb,cb)      — CUDA line ~120
            for k (depth tiles, innermost)             — CUDA line ~122
              q_frag = load Q[rb, k]                   — CUDA line ~132
              k_frag = load K[cb, k]                   — CUDA line ~133
              c_frag += q_frag @ k_frag.T * scale      — CUDA line ~155 (mma.m16n8k16)
            softmax online update on c_frag            — CUDA lines ~174-248
            for co in 0..D/8:                          — CUDA line ~265
              v_frag = load V[cb, co]                  — CUDA line ~268
              o_reg[rb, co] = rescale*o + P @ v_frag   — CUDA lines ~283-293
    """
    active_q_rows  = Q_smem.shape[0]
    active_kv_cols = K_smem.shape[0]
    D              = Q_smem.shape[1]

    scale       = 1.0 / np.sqrt(float(D))
    k_tiles     = D  // k_tile_size
    c_row_tiles = (active_q_rows  + Br     - 1) // Br
    c_col_tiles = (active_kv_cols + Bc_col - 1) // Bc_col
    co_tiles    = D  // v_tile_cols               # CUDA: DHeadDim / 8

    for rb in range(c_row_tiles):                                  # CUDA: warp_id
        for cb in range(c_col_tiles):                              # CUDA line ~115

            q_row_start  = rb * Br
            kv_col_start = cb * Bc_col
            n_q  = min(Br,     active_q_rows  - q_row_start)
            n_kv = min(Bc_col, active_kv_cols - kv_col_start)

            # c_frag reset — CUDA line ~120: zeros before k-accumulation loop
            c_frag = np.zeros((Br, Bc_col), dtype=np.float32)

            for k in range(k_tiles):                               # CUDA line ~122
                q_frag = smem_load_frag_a(Q_smem, rb, k)          # [Br, k_tile_size]
                k_frag = smem_load_frag_b(K_smem, cb, k)          # [Bc_col, k_tile_size]
                c_frag += (q_frag @ k_frag.T) * scale             # [Br, Bc_col] mma.m16n8k16

            # Softmax fires here after ALL k-tiles accumulated — CUDA lines ~174-248

            # Mask out-of-bounds rows/cols → -inf  — CUDA lines ~178-193
            scores = np.full((Br, Bc_col), -np.inf, dtype=np.float32)
            scores[:n_q, :n_kv] = c_frag[:n_q, :n_kv]

            # Per-row max + merge with running mi_reg  — CUDA lines ~206-221
            row_max = scores.max(axis=1)
            mi_old  = np.full(Br, -np.inf, dtype=np.float32)
            mi_old[:n_q] = mi_reg[q_row_start:q_row_start + n_q]
            mi_new  = np.maximum(mi_old, row_max)

            # Softmax numerators  — CUDA lines ~225-227
            P = np.exp(scores - mi_new[:, None])                   # [Br, Bc_col]

            # Row-sum reduction + li_reg/mi_reg update in registers  — CUDA lines ~231-248
            rescale = np.exp(mi_old - mi_new)                      # [Br]
            row_sum = P.sum(axis=1)
            li_old  = np.zeros(Br, dtype=np.float32)
            li_old[:n_q] = li_reg[q_row_start:q_row_start + n_q]
            li_new  = li_old * rescale + row_sum

            mi_reg[q_row_start:q_row_start + n_q] = mi_new[:n_q]
            li_reg[q_row_start:q_row_start + n_q] = li_new[:n_q]

            # cvt.rn.f16x2 packing modelled as plain float  — CUDA lines ~255-258
            p_a = P   # [Br, Bc_col]

            # co loop: o_reg += rescale*o_old + P @ V_tile  — CUDA lines ~265-300
            for co in range(co_tiles):                             # CUDA line ~265
                V_tile = smem_load_frag_b8_trans(V_smem, cb, co)  # [Bc_col, v_tile_cols]

                d_start = co * v_tile_cols
                d_end   = d_start + v_tile_cols

                O_slice = o_reg[q_row_start:q_row_start + n_q, d_start:d_end].copy()
                O_slice *= rescale[:n_q, None]                         # CUDA lines ~283-284
                O_slice += p_a[:n_q, :n_kv] @ V_tile[:n_kv, :]        # CUDA lines ~288-293

                o_reg[q_row_start:q_row_start + n_q, d_start:d_end] = O_slice


# ── flash_attn_kernel ─────────────────────────────────────────────────────────
def flash_attn_kernel(
    Q:             np.ndarray,    # [B, H_q,  S, D] fp32
    K:             np.ndarray,    # [B, H_kv, S, D] fp32
    V:             np.ndarray,    # [B, H_kv, S, D] fp32
    kv_block_rows: int = 128,
) -> np.ndarray:
    """
    Python model of flash_attn_kernel() in flash_attn.cu.

    CUDA grid  : (B, H_q, ceil(S/kFlashBr))  — outer loops below
    CUDA block : kFlashWarps*32 threads       — one warp_flash_attn call per kv_tile

    v2 register model: mi_reg/li_reg/o_reg are initialized per Q tile, accumulated
    across all KV tiles in warp_flash_attn, then written to HBM once.
    This mirrors the CUDA kernel where O/Mi/Li live in registers throughout the KV loop.
    """
    B, H_q, S, D = Q.shape
    H_kv      = K.shape[1]
    gqa_ratio = H_q // H_kv

    O = np.zeros((B, H_q, S, D), dtype=np.float32)

    q_tiles  = (S + Br - 1) // Br
    kv_tiles = (S + kv_block_rows - 1) // kv_block_rows

    for b in range(B):                                      # CUDA blockIdx.x
        for h_q in range(H_q):                              # CUDA blockIdx.y
            h_kv = h_q // gqa_ratio                         # GQA  — CUDA line ~359

            for q_tile in range(q_tiles):                   # CUDA blockIdx.z
                q_row_start = q_tile * Br
                q_rows      = min(Br, S - q_row_start)

                # smem_q  — CUDA lines ~401-408
                smem_q          = np.zeros((Br, D), dtype=np.float32)
                smem_q[:q_rows] = Q[b, h_q, q_row_start:q_row_start + q_rows]

                # Register accumulators, init'd once per Q tile  — CUDA lines ~450-455
                mi_reg = np.full(q_rows, -np.inf, dtype=np.float32)
                li_reg = np.zeros(q_rows,          dtype=np.float32)
                o_reg  = np.zeros((q_rows, D),     dtype=np.float32)

                for kv_tile in range(kv_tiles):             # CUDA line ~462
                    kv_start = kv_tile * kv_block_rows
                    kv_rows  = min(kv_block_rows, S - kv_start)

                    # smem_k / smem_v (cp.async double-buffered in CUDA) — CUDA lines ~424-439
                    smem_k           = np.zeros((kv_block_rows, D), dtype=np.float32)
                    smem_v           = np.zeros((kv_block_rows, D), dtype=np.float32)
                    smem_k[:kv_rows] = K[b, h_kv, kv_start:kv_start + kv_rows]
                    smem_v[:kv_rows] = V[b, h_kv, kv_start:kv_start + kv_rows]

                    warp_flash_attn(
                        smem_q[:q_rows], smem_k[:kv_rows], smem_v[:kv_rows],
                        mi_reg, li_reg, o_reg,
                    )

                # Single HBM write after all KV tiles  — CUDA lines ~482-501
                O_block = O[b, h_q, q_row_start:q_row_start + q_rows]
                O_block[:] = o_reg
                O_block /= li_reg[:, None]                  # normalize: CUDA lines ~499-501

    return O


# ── Reference: correct scaled dot-product attention ───────────────────────────
def reference_attn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Standard SDPA in fp32; correctness baseline."""
    B, H_q, S, D = Q.shape
    H_kv      = K.shape[1]
    gqa_ratio = H_q // H_kv
    scale     = 1.0 / np.sqrt(float(D))
    O         = np.zeros_like(Q)
    for b in range(B):
        for h in range(H_q):
            h_kv   = h // gqa_ratio
            scores = Q[b, h] @ K[b, h_kv].T * scale
            scores -= scores.max(axis=-1, keepdims=True)
            P      = np.exp(scores)
            P     /= P.sum(axis=-1, keepdims=True)
            O[b, h] = P @ V[b, h_kv]
    return O


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    for D in (16, 64, 128):
        B, S, H_q, H_kv = 1, 16, 2, 1
        Q = np.random.randn(B, H_q, S, D).astype(np.float32)
        K = np.random.randn(B, H_kv, S, D).astype(np.float32)
        V = np.random.randn(B, H_kv, S, D).astype(np.float32)
        got = flash_attn_kernel(Q, K, V, kv_block_rows=S)
        ref = reference_attn(Q, K, V)
        err = float(np.abs(got - ref).max())
        status = "PASS" if err < 1e-4 else f"FAIL  max_err={err:.6f}"
        print(f"D={D:3d}  max_abs_err={err:.6f}  {status}")
