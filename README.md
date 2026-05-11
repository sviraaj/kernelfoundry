# KernelFoundry

**Custom CUDA kernels for AI inference — written from scratch using Tensor Core MMA instructions and PTX.**

Each kernel ships with a NumPy reference model that mirrors the exact tile geometry and algorithm — making it straightforward to verify correctness and reason about tiling changes without touching CUDA.

> Primary target: **RTX 5070 (Blackwell sm_120)** · CUDA 12.9 · Minimum sm_80 (Ampere)

---

## Kernels

| Kernel | Status | Features |
|--------|:------:|----------|
| Flash Attention | ✅ v1 | GQA / MQA, m16n8k16 + m16n8k8 MMA, online softmax, decode (Q_len=1) |
| RMSNorm | 🔲 | — |
| RoPE | 🔲 | — |
| SwiGLU | 🔲 | — |

---

## Flash Attention — v1 Numbers

**RTX 5070 · H_q=32, H_kv=8 (GQA 4:1), D=128, fp16 in / fp32 out**

**Prefill** (Q_len = KV_len):

| Seq len | Latency  | TFLOPS |
|--------:|---------:|-------:|
| 512     | 3.0 ms   | 1.42   |
| 1024    | 10.4 ms  | 1.65   |
| 2048    | 40.8 ms  | 1.69   |
| 4096    | 162.6 ms | 1.69   |

**Decode** (Q_len=1, full KV cache):

| KV cache len | Latency  |
|-------------:|---------:|
| 512          | 0.32 ms  |
| 1024         | 0.59 ms  |
| 2048         | 1.12 ms  |
| 4096         | 2.18 ms  |

*v1 baseline — O accumulator currently round-trips to HBM per tile. Next: keep O in registers across the KV sweep.*

Compare with PyTorch FA2 (SDPA): `python reference_benchmarks/bench_flash_attn2.py`

---

## Quick Start

```bash
# Configure (Blackwell)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120

# Build everything
cmake --build build -j$(nproc)

# Correctness tests (18 shapes — GQA, MQA, decode, variable D)
cd build && ctest --output-on-failure

# C++ GPU benchmark
./build/tests/bench_flash_attn

# PyTorch FA2 reference benchmark
python reference_benchmarks/bench_flash_attn2.py

# NumPy model smoke test (no GPU needed)
python kernel_models/flash_attn.py
```

For Ampere or a custom CUDA path:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0
```

---

## Repository Layout

```
kernelfoundry/
├── kernels/                      # CUDA kernel implementations
│   └── flash_attn.cu             #   Flash Attention — MMA + online softmax
│
├── kernel_models/                # NumPy reference models (algorithm verification)
│   └── flash_attn.py             #   Mirrors flash_attn.cu tile-for-tile
│
├── reference_benchmarks/         # Industry baseline benchmarks
│   └── bench_flash_attn2.py      #   PyTorch FA2 (SDPA) — same shapes as C++ bench
│
├── tests/                        # Correctness tests + C++ GPU benchmarks
│   ├── test_flash_attn.cu        #   18 end-to-end test cases
│   ├── bench_flash_attn.cu       #   Latency / TFLOPS / GB/s via CUDA Events
│   ├── test_utils.h
│   └── CMakeLists.txt            #   Includes ncu profiling targets
│
├── docs/                         # Algorithm notes and design docs
│   ├── flash_attn.md             #   Tile geometry, smem layout, ncu guide
│   └── flash_attn_v1.txt         #   Original design scratch notes
│
├── tools/
│   └── run_ptx.c                 # CUDA Driver API PTX/cubin runner
│
├── CMakeLists.txt
└── LICENSE
```

---

## GPU Profiling

ncu targets are built into CMake. Run from `build/` with `sudo`:

```bash
sudo cmake --build . --target ncu_stalls       # occupancy + stall breakdown
sudo cmake --build . --target ncu_bandwidth    # HBM / L2 / smem byte counts
sudo cmake --build . --target ncu_tensorcore   # HMMA pipe utilization
sudo cmake --build . --target ncu_roofline     # roofline chart → reports/
sudo cmake --build . --target ncu_full         # full report → reports/

sudo cmake --build . --target ncu_sdpa_stalls  # same metrics for PyTorch FA2
```

See [docs/flash_attn.md](docs/flash_attn.md) for metric interpretation and the
`smsp__warps_eligible` gotcha.

---

## References

- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [NVIDIA PTX ISA — `mma` instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)

---

## License

MIT — see [LICENSE](LICENSE).
