# KernelFoundry

**Custom CUDA kernels for AI inference — written from scratch using Tensor Core MMA instructions and PTX.**

Each kernel ships with a NumPy reference model that mirrors the exact tile geometry and algorithm — making it straightforward to verify correctness and reason about tiling changes without touching CUDA.

> Primary target: **RTX 5070 (Blackwell sm_120)** · CUDA 12.9 · Minimum sm_80 (Ampere)

---

## Kernels

| Kernel | Status | Features |
|--------|:------:|----------|
| Flash Attention | ✅ v2 | GQA / MQA, m16n8k16 + m16n8k8 MMA, online softmax, decode (Q_len=1), cp.async KV double-buffering, O/Mi/Li in registers, template-D (zero register spill), XOR smem swizzle |
| RMSNorm | 🔲 | — |
| RoPE | 🔲 | — |
| SwiGLU | 🔲 | — |

---

## Flash Attention — Performance

**RTX 5070 · Blackwell sm_120 · H_q=32, H_kv=8 (GQA 4:1), D=128, fp16 in / fp32 out**

### Optimization history

Each row is a single incremental change on top of the previous.

| Stage | Optimization | Prefill S=1024 | Prefill S=4096 | Decode KV=4096 | vs v1 |
|------:|---|---:|---:|---:|---:|
| v1 | Baseline — synchronous KV loads, O written to HBM each KV tile | 10.46 ms · 1.64T | 158.2 ms · 1.74T | — | 1.0× |
| v2a | **cp.async KV double-buffering** — async copy overlaps HBM→smem with MMA | 3.34 ms · 5.14T | 46.0 ms · 5.97T | 0.55 ms | 3.1× |
| v2b | **O / Mi / Li in registers** — accumulators stay in RF across all KV tiles | 3.25 ms · 5.29T | 44.5 ms · 6.18T | 0.57 ms | 3.2× |
| v2c | **Template-D + o_frag unroll** — eliminates 544-byte register spill to local memory | 1.79 ms · 9.62T | 25.7 ms · 10.7T | 0.44 ms | 5.9× |
| v2d | **XOR smem swizzle** — eliminates 8-way ldmatrix bank conflicts on K/V reads | **1.03 ms · 16.7T** | **14.3 ms · 19.3T** | **0.30 ms** | **10.1×** |

### vs PyTorch SDPA (FA2 backend)

Same hardware, same shapes. SDPA uses the FlashAttention-2 backend with Blackwell-optimized kernels (WGMMA, TMA).

**Prefill** (Q_len = KV_len, B=1):

| Seq len | Ours | SDPA | Gap |
|--------:|-----:|-----:|----:|
| 512  |  0.305 ms · 14.1T |  0.091 ms · 47.1T | 3.4× |
| 1024 |  1.028 ms · 16.7T |  0.332 ms · 51.8T | 3.1× |
| 2048 |  3.866 ms · 17.8T |  1.208 ms · 56.9T | 3.2× |
| 4096 | 14.279 ms · 19.3T |  4.810 ms · 57.1T | 3.0× |

**Decode** (Q_len=1, B=1, full KV cache):

| KV cache len | Ours | SDPA | Gap |
|-------------:|-----:|-----:|----:|
|  512 | 0.040 ms | 0.023 ms | 1.7× |
| 1024 | 0.078 ms | 0.029 ms | 2.7× |
| 2048 | 0.165 ms | 0.049 ms | 3.4× |
| 4096 | 0.303 ms | 0.119 ms | 2.5× |

Run the reference benchmark: `python reference_benchmarks/bench_flash_attn2.py`

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
