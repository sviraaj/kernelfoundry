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


if __name__ == "__main__":
    main()
