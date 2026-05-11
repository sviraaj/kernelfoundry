# Flash Attention — Implementation Notes

## Tile Geometry

```
Tensor  Shape in smem        MMA instruction
──────  ──────────────────   ──────────────────────────────────────
Q tile  [Br=16,  k=16] fp16  mma.sync.aligned.m16n8k16  (A operand)
K tile  [Bc=8,   k=16] fp16  mma.sync.aligned.m16n8k16  (B operand)
V tile  [Bc=8,   n=8]  fp16  mma.sync.aligned.m16n8k8   (B operand, ldmatrix.trans)
P tile  [Br=16,  Bc=8] fp32  softmax numerators, packed fp16 for PV MMA
O tile  [Br=16,  n=8]  fp32  mma.sync.aligned.m16n8k8   (C/D accumulator)
```

Tile constants:

| Constant          | Value              | Role                                  |
|-------------------|--------------------|---------------------------------------|
| `kFlashWarps`     | 4                  | warps per CUDA block                  |
| `kFlashBr`        | `kFlashWarps × 16` | Q rows per block (= 64)               |
| `kFlashBcDefault` | 128                | KV rows per smem block (auto-tuned)   |
| `kFlashBcMin`     | 8                  | minimum KV block size                 |

## Shared Memory Layout

```
smem_raw:
  [0              .. kFlashBr * D)              — smem_q  (Q tile, fp16)
  [kFlashBr*D     .. (kFlashBr + Bc)*D)         — smem_k  (K block, fp16)
  [(kFlashBr+Bc)*D .. (kFlashBr + 2*Bc)*D)      — smem_v  (V block, fp16)

smem_bytes = (kFlashBr + 2 * kv_block_rows) * D * 2

D=128, kv_block_rows=128:  (64 + 256) * 128 * 2 = 81,920 B  (~80 KB)
D=256, kv_block_rows=64:   (64 + 128) * 256 * 2 = 98,304 B  (~96 KB)
```

With 80 KB smem per block, the RTX 5070 fits 1 block/SM (shared mem limit ~128 KB).
`kFlashWarps=4` raises occupancy from 1 warp to 4 warps per SM without adding blocks.

## Multi-Warp Design (warp-per-Q-row-tile)

Each CUDA block covers `kFlashBr = 64` Q rows. Warps partition those rows with
no inter-warp communication needed — each warp owns independent Mi/Li/Oi rows:

```
warp 0 → rb = 0, 4, 8, ...   (rows  0–15,  64–79, ...)
warp 1 → rb = 1, 5, 9, ...   (rows 16–31,  80–95, ...)
warp 2 → rb = 2, 6, 10, ...  (rows 32–47,  96–111, ...)
warp 3 → rb = 3, 7, 11, ...  (rows 48–63, 112–127, ...)
```

All warps read the same K/V smem block (reads never conflict), so no barriers
are needed inside the attention loop.

## Loop Structure

```
flash_attn_kernel  (grid: B × H_q × ceil(q_len / kFlashBr)):
  load smem_q  [kFlashBr, D]  ← once per block

  for each kv_tile  (streams K/V from HBM into smem_k, smem_v):
    __syncthreads()

    warp_flash_attn  (each warp: rb = warp_id, warp_id+4, ...):
      for each rb tile:
        c_frag = 0                                  ← reset once per (rb, cb)
        for each k tile (depth dimension D):
          c_frag += Q[rb,k] @ K[cb,k].T * scale    ← mma.m16n8k16

        row_max  = reduce_max(c_frag)
        mi_new   = max(mi_old, row_max)
        P        = exp(c_frag − mi_new)
        li_new   = li_old * exp(mi_old − mi_new) + rowsum(P)
        write Mi, Li → HBM                          ← current bottleneck
        rescale = exp(mi_old − mi_new)

        for each co tile (D output columns):
          O[rb,co] = rescale * O[rb,co] + P @ V[cb,co]   ← mma.m16n8k8
        write O → HBM                               ← current bottleneck

    __syncthreads()

  O /= Li   (final normalization)
```

## Grid and Block

```cpp
dim3 grid (B, H_q, (q_len + kFlashBr - 1) / kFlashBr);
dim3 block(kFlashWarps * 32);   // 128 threads = 4 warps
size_t smem = (kFlashBr + 2 * kv_block_rows) * D * sizeof(uint16_t);
```

## GQA / MQA

When `H_kv < H_q`, each K/V head is shared by `H_q / H_kv` query heads.
The kernel uses `head_q % H_kv` to index into K/V, so all GQA ratios are supported
with no memory duplication.

## Deviations from Flash Attention 2

| Deviation | Correct? | Notes |
|-----------|----------|-------|
| No causal mask | **Wrong for decoder** | Required before autoregressive use |
| O round-tripped to HBM per (rb, cb) | Correct, slow | FA2 keeps O in registers across full KV sweep |
| Mi/Li round-tripped to HBM per (rb, cb) | Correct, slow | FA2 keeps in registers |
| fp32 output (vs fp16) | Intentional | 2× write bandwidth vs FA2 |
| No async K/V prefetch | Correct, slow | FA2 uses `cp.async` to pipeline loads |

## ncu Profiling Targets

Built into CMake — run from `build/` with `sudo`:

| Target | What it measures |
|--------|-----------------|
| `ncu_stalls` | Occupancy + stall breakdown (stdout) |
| `ncu_bandwidth` | HBM / L2 / smem byte counts |
| `ncu_tensorcore` | HMMA pipe utilization % |
| `ncu_roofline` | Roofline chart → `build/reports/ncu_roofline.ncu-rep` |
| `ncu_full` | Full section report → `build/reports/ncu_full.ncu-rep` |
| `ncu_sdpa_stalls` | Same stall metrics for PyTorch FA2 kernel |
| `ncu_sdpa_full` | Full FA2 report for direct comparison |

**Reading the smsp warp metric:** `smsp__warps_eligible = 1.0` does NOT mean one
warp is active — it means 1 warp per SM sub-partition (SMSP). Each SM has 4 SMSPs,
so `1.0` on this metric = 4 warps total per SM, which is correct for `kFlashWarps=4`.
Use `sm__warps_active.avg.per_cycle_active` for the absolute SM-level count.

## Performance History

RTX 5070, B=1 S=1024 H_q=32 H_kv=8 D=128:

| Change | Latency | TFLOPS | vs FA2 |
|--------|--------:|-------:|-------:|
| Baseline (`kFlashWarps=1`) | 74 ms | 0.23 | ~215× |
| `kFlashWarps=4` (v1) | 10.4 ms | 1.65 | ~31× |

The multi-warp change raised SM occupancy from ~2% to ~8% (4 warps × 1 block/SM)
and delivered a 7× speedup. The remaining gap is primarily O/Mi/Li HBM round-trips.
