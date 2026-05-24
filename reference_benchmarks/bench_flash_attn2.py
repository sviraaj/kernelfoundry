#!/usr/bin/env python3
"""bench_flash_attn2.py — PyTorch Flash Attention 2 (SDPA) reference benchmark.

Benchmarks the same shapes as bench_flash_attn (C++ kernel) for direct
comparison. Uses torch.nn.functional.scaled_dot_product_attention with the
FLASH_ATTENTION backend.

Metrics:
  latency   ms/iter  — CUDA event timing, warmup excluded
  TFLOPS             — 4·B·H_q·q_len·kv_len·D (QKᵀ + PV)
  bandwidth GB/s     — Q+K+V reads (fp16) + O write (fp16)
                       NOTE: FA2 output is fp16; our C++ kernel writes fp32.

Decode mode (q_len=1):
  Q shape  [B, H_q,  1,      D]  — single new token per sequence
  KV shape [B, H_kv, kv_len, D]  — full KV cache

Sections:
  1. Baseline prefill + decode  (same shapes as C++ bench)
  2. Persistent KV cache decode — pre-allocated [B, H_kv, max_len, D] buffer,
     decode at growing cache checkpoints; models steady-state inference.
  3. Paged KV attention overhead — block-table index_select gather + SDPA vs
     contiguous SDPA; shows the overhead that a specialized paged-attn kernel
     (FlashInfer / vLLM) eliminates.

Run:
    /home/sviraaj/projects/AI/.venv/bin/python reference_benchmarks/bench_flash_attn2.py
"""

import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

WARMUP = 5
ITERS  = 50
DTYPE  = torch.float16
DEVICE = "cuda"


def run_bench(B, kv_len, H_q, H_kv, D, q_len=None):
    """
    q_len=None  → q_len = kv_len  (prefill / self-attention)
    q_len=1     → single-token decode against a KV cache of length kv_len
    """
    if q_len is None:
        q_len = kv_len

    Q = torch.randn(B, H_q,  q_len,  D, dtype=DTYPE, device=DEVICE)
    K = torch.randn(B, H_kv, kv_len, D, dtype=DTYPE, device=DEVICE)
    V = torch.randn(B, H_kv, kv_len, D, dtype=DTYPE, device=DEVICE)

    # PyTorch's FLASH_ATTENTION path requires H_q == H_kv; expand K/V for GQA.
    # Bandwidth is computed against original (non-expanded) KV size.
    gqa = H_kv != H_q
    K_in = K.repeat_interleave(H_q // H_kv, dim=1) if gqa else K
    V_in = V.repeat_interleave(H_q // H_kv, dim=1) if gqa else V
    label = "FA2-GQA-expanded" if gqa else "FA2"

    decode = (q_len == 1 and kv_len > 1)
    shape_str = (f"B={B:<2d} Q=1    KV={kv_len:<4d} H_q={H_q:<2d} H_kv={H_kv:<2d} D={D:<3d}"
                 if decode else
                 f"B={B:<2d} S={kv_len:<4d}       H_q={H_q:<2d} H_kv={H_kv:<2d} D={D:<3d}")

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            for _ in range(WARMUP):
                F.scaled_dot_product_attention(Q, K_in, V_in)
            torch.cuda.synchronize()

            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            for _ in range(ITERS):
                F.scaled_dot_product_attention(Q, K_in, V_in)
            ev_e.record()
            torch.cuda.synchronize()

        ms = ev_s.elapsed_time(ev_e) / ITERS
    except RuntimeError as exc:
        print(f"  {shape_str} | SKIP ({exc})")
        return

    # FLOPs: QKᵀ + PV = 2 × (2·B·H_q·q_len·kv_len·D MACs)
    flops  = 4.0 * B * H_q * q_len * kv_len * D
    tflops = flops / (ms * 1e9)

    # Bandwidth: original (non-expanded) KV sizes
    Q_n  = B * H_q  * q_len  * D
    KV_n = B * H_kv * kv_len * D
    nbytes = 2 * Q_n + 2 * KV_n + 2 * KV_n + 2 * Q_n   # fp16 throughout
    bw_gbs = nbytes / (ms * 1e6)

    print(f"  {shape_str} | {ms:7.3f} ms | {tflops:7.3f} TFLOPS | {bw_gbs:7.1f} GB/s  [{label}]")


def _time_fn(fn, warmup=WARMUP, iters=ITERS):
    """Warmup + CUDA-event timing. Returns ms/iter, or None on RuntimeError."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        for _ in range(iters):
            fn()
        ev_e.record()
        torch.cuda.synchronize()
        return ev_s.elapsed_time(ev_e) / iters
    except RuntimeError as exc:
        return exc


def run_kvcache_bench(B, max_kv_len, H_q, H_kv, D,
                      checkpoints=(128, 256, 512, 1024, 2048, 4096)):
    """Decode (Q_len=1) against a persistent pre-allocated KV cache.

    A single [B, H_kv, max_kv_len, D] buffer is allocated once (models a
    real inference server that pre-allocates max context).  At each checkpoint
    kv_len we take a slice [:, :, :kv_len, :] and measure decode attention
    latency.  This shows how latency scales as the context window fills.
    """
    K_cache = torch.zeros(B, H_kv, max_kv_len, D, dtype=DTYPE, device=DEVICE)
    V_cache = torch.zeros(B, H_kv, max_kv_len, D, dtype=DTYPE, device=DEVICE)
    Q       = torch.randn(B, H_q, 1, D, dtype=DTYPE, device=DEVICE)
    gqa     = H_q != H_kv
    ratio   = H_q // H_kv

    for kv_len in [c for c in checkpoints if c <= max_kv_len]:
        K_v = K_cache[:, :, :kv_len, :]
        V_v = V_cache[:, :, :kv_len, :]
        K_in = K_v.repeat_interleave(ratio, dim=1) if gqa else K_v
        V_in = V_v.repeat_interleave(ratio, dim=1) if gqa else V_v

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            result = _time_fn(lambda: F.scaled_dot_product_attention(Q, K_in, V_in))

        shape = f"B={B:<2d} Q=1    KV={kv_len:<5d} H_q={H_q:<2d} H_kv={H_kv:<2d} D={D:<3d}"
        if isinstance(result, Exception):
            print(f"  {shape} | SKIP ({result})")
            continue

        ms = result
        flops  = 4.0 * B * H_q * kv_len * D
        tflops = flops / (ms * 1e9)
        # Q read + KV read (H_kv, not expanded) + O write — all fp16
        nbytes = 2 * (B * H_q * D + 2 * B * H_kv * kv_len * D + B * H_q * D)
        bw_gbs = nbytes / (ms * 1e6)
        print(f"  {shape} | {ms:7.3f} ms | {tflops:6.3f} TFLOPS | {bw_gbs:7.1f} GB/s  [persistent-cache]")


def run_paged_kv_bench(B, kv_len, H_q, H_kv, D, block_size=16):
    """Paged KV attention overhead vs contiguous SDPA.

    Layout:
      K_pool / V_pool : [pool_blocks, H_kv, block_size, D]  (fragmented physical pages)
      block_ids       : [num_blocks]  int64  (random permutation — worst-case fragmentation)

    Gather step: K_pool[block_ids] → reshape → [B, H_kv, kv_len, D]
    This is what you pay *without* a specialised paged-attention kernel.
    A dedicated kernel (FlashInfer / vLLM) fuses this gather into the attention
    loop and eliminates the intermediate allocation.

    Reports:
      contiguous : baseline SDPA on a contiguous [B, H_kv, kv_len, D] tensor
      paged      : gather (index_select) + SDPA — naive paged path
      overhead % : extra latency of naive paging vs contiguous
    """
    num_blocks = (kv_len + block_size - 1) // block_size
    padded     = num_blocks * block_size
    pool_size  = num_blocks * 2          # 2× pool → random block assignment
    gqa        = H_q != H_kv
    ratio      = H_q // H_kv

    K_pool  = torch.randn(pool_size, H_kv, block_size, D, dtype=DTYPE, device=DEVICE)
    V_pool  = torch.randn(pool_size, H_kv, block_size, D, dtype=DTYPE, device=DEVICE)
    blk_ids = torch.randperm(pool_size, device=DEVICE)[:num_blocks]

    Q       = torch.randn(B, H_q, 1, D, dtype=DTYPE, device=DEVICE)

    # Contiguous baseline
    K_cont = torch.randn(B, H_kv, kv_len, D, dtype=DTYPE, device=DEVICE)
    V_cont = torch.randn(B, H_kv, kv_len, D, dtype=DTYPE, device=DEVICE)
    K_cin  = K_cont.repeat_interleave(ratio, dim=1) if gqa else K_cont
    V_cin  = V_cont.repeat_interleave(ratio, dim=1) if gqa else V_cont

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        r_cont = _time_fn(lambda: F.scaled_dot_product_attention(Q, K_cin, V_cin))

    # Paged gather + SDPA
    def paged_step():
        # Gather: [num_blocks, H_kv, block_size, D]
        Kg = K_pool[blk_ids]
        Vg = V_pool[blk_ids]
        # → [B, H_kv, kv_len, D]  (contiguous copy required before SDPA)
        Kc = Kg.permute(1, 0, 2, 3).contiguous().view(H_kv, padded, D)[:, :kv_len, :] \
               .unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        Vc = Vg.permute(1, 0, 2, 3).contiguous().view(H_kv, padded, D)[:, :kv_len, :] \
               .unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        Ki = Kc.repeat_interleave(ratio, dim=1) if gqa else Kc
        Vi = Vc.repeat_interleave(ratio, dim=1) if gqa else Vc
        return F.scaled_dot_product_attention(Q, Ki, Vi)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        r_paged = _time_fn(paged_step)

    shape = f"B={B:<2d} Q=1    KV={kv_len:<5d} H_q={H_q:<2d} H_kv={H_kv:<2d} D={D:<3d} bs={block_size}"
    if isinstance(r_cont, Exception) or isinstance(r_paged, Exception):
        err = r_cont if isinstance(r_cont, Exception) else r_paged
        print(f"  {shape} | SKIP ({err})")
        return

    overhead = (r_paged / r_cont - 1.0) * 100.0
    print(f"  {shape} | {r_cont:6.3f} ms [contiguous] | {r_paged:6.3f} ms [paged-gather] | +{overhead:5.1f}% overhead")


def main():
    if not torch.cuda.is_available():
        sys.exit("No CUDA device found.")

    name = torch.cuda.get_device_name(0)
    print(f"=== Flash Attention 2 (PyTorch SDPA) Benchmark — {name} ===")
    print(f"    warmup={WARMUP}, iters={ITERS}, dtype=fp16")
    print(f"    NOTE: bandwidth uses fp16 output; C++ bench uses fp32 output.")
    print(f"  {'B':<4} {'S':<5} {'H_q':<5} {'H_kv':<6} {'D':<4}"
          f" | {'latency':<10} | {'throughput':<14} | {'bandwidth'}")

    print("\n[decode: Q_len=1 (single new token), KV_len=full cache]")
    run_bench(1,   512, 32,  8, 128, q_len=1)
    run_bench(1,  1024, 32,  8, 128, q_len=1)
    run_bench(1,  2048, 32,  8, 128, q_len=1)
    run_bench(1,  4096, 32,  8, 128, q_len=1)
    run_bench(8,   512, 32,  8, 128, q_len=1)   # batched decode
    run_bench(8,  1024, 32,  8, 128, q_len=1)

    print("\n[short-prefill: Q_len=KV_len, small S]")
    run_bench(1,   64, 32,  8, 128)
    run_bench(1,  128, 32,  8, 128)
    run_bench(1,  256, 32,  8, 128)
    run_bench(8,   64, 32,  8, 128)

    print("\n[prefill: large S]")
    run_bench(1,  512, 32,  8, 128)
    run_bench(1, 1024, 32,  8, 128)
    run_bench(1, 2048, 32,  8, 128)
    run_bench(1, 4096, 32,  8, 128)

    print("\n[head dim sweep: B=1 S=1024 H_q=32 H_kv=8]")
    run_bench(1, 1024, 32,  8,  64)
    run_bench(1, 1024, 32,  8, 128)
    run_bench(1, 1024, 32,  8, 256)

    print("\n[MQA: H_kv=1]")
    run_bench(1,  512, 32,  1, 128)
    run_bench(1, 1024, 32,  1, 128)
    run_bench(1,  512, 32,  1, 128, q_len=1)
    run_bench(1, 1024, 32,  1, 128, q_len=1)

    # ------------------------------------------------------------------
    # Section 2: Persistent KV cache decode
    # Pre-allocate max-context buffer once; measure decode at growing kv_len.
    # Models a real inference server: no per-step allocation, just a slice.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== Persistent KV Cache Decode (pre-allocated max-context buffer) ===")
    print("=" * 80)
    print("  Pre-allocated once at max_kv_len; each row slices [:kv_len] of that buffer.")
    print("  Bandwidth counts original H_kv KV size (not GQA-expanded).")
    print(f"  {'B':<4} {'shape':<40} | {'latency':<10} | {'TFLOPS':<8} | {'bandwidth'}")

    print("\n[GQA 4:1 — Llama-style H_q=32 H_kv=8 D=128, max_kv=4096]")
    run_kvcache_bench(1, 4096, 32, 8, 128,
                      checkpoints=(128, 256, 512, 1024, 2048, 4096))

    print("\n[GQA 4:1 — batched B=8, max_kv=2048]")
    run_kvcache_bench(8, 2048, 32, 8, 128,
                      checkpoints=(256, 512, 1024, 2048))

    print("\n[MQA H_kv=1, max_kv=4096]")
    run_kvcache_bench(1, 4096, 32, 1, 128,
                      checkpoints=(128, 256, 512, 1024, 2048, 4096))

    print("\n[head dim D=256, max_kv=2048]")
    run_kvcache_bench(1, 2048, 32, 8, 256,
                      checkpoints=(256, 512, 1024, 2048))

    # ------------------------------------------------------------------
    # Section 3: Paged KV attention overhead
    # Compare contiguous SDPA vs block-table gather + SDPA.
    # The "overhead %" is what a fused paged-attn kernel (FlashInfer/vLLM)
    # eliminates by doing the page lookup inside the attention loop.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== Paged KV Attention Overhead (gather + SDPA vs contiguous SDPA) ===")
    print("=" * 80)
    print("  Layout: K/V_pool [2×num_blocks, H_kv, block_size, D], random block_ids.")
    print("  Naive path: index_select gather → contiguous buffer → SDPA.")
    print("  Overhead = what a fused paged-attn kernel (FlashInfer/vLLM) eliminates.")
    print(f"  {'shape + block_size':<55} | {'contiguous':>12} | {'paged-gather':>13} | overhead")

    print("\n[decode Q=1, GQA 4:1, D=128 — block_size sweep]")
    for bs in (16, 32, 64):
        run_paged_kv_bench(1,  512, 32, 8, 128, block_size=bs)
    for bs in (16, 32, 64):
        run_paged_kv_bench(1, 1024, 32, 8, 128, block_size=bs)
    for bs in (16, 32, 64):
        run_paged_kv_bench(1, 2048, 32, 8, 128, block_size=bs)
    for bs in (16, 32, 64):
        run_paged_kv_bench(1, 4096, 32, 8, 128, block_size=bs)

    print("\n[decode Q=1, MQA H_kv=1, D=128, block_size=16]")
    run_paged_kv_bench(1,  512, 32, 1, 128, block_size=16)
    run_paged_kv_bench(1, 1024, 32, 1, 128, block_size=16)
    run_paged_kv_bench(1, 2048, 32, 1, 128, block_size=16)

    print("\n[batched B=4, GQA 4:1, D=128, block_size=16]")
    run_paged_kv_bench(4,  512, 32, 8, 128, block_size=16)
    run_paged_kv_bench(4, 1024, 32, 8, 128, block_size=16)
    run_paged_kv_bench(4, 2048, 32, 8, 128, block_size=16)


if __name__ == "__main__":
    main()
