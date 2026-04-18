# warpforge

Custom CUDA kernel library for LLM inference, built from scratch using Tensor Core
MMA instructions. Targets Llama-style transformer architectures on Blackwell (sm_120)
and Ampere (sm_80+).

> **Status:** Flash Attention kernel is functionally correct and benchmarked.
> Core inference stubs (rmsnorm, rope, swiglu, sampling) remain to be implemented.

---

## Architecture Target

Llama-3 8B equivalent configuration.

| Parameter        | Value     |
|------------------|-----------|
| Layers           | 32        |
| Model dimension  | 4096      |
| Attention heads  | 32        |
| KV heads (GQA)   | 8         |
| Head dimension   | 128       |
| FFN hidden       | 14336     |
| Max context      | 8192      |
| RoPE theta       | 1,000,000 |
| Vocab size       | 32,000    |

---

## Flash Attention Kernel

`kernels/flash_attn.cu` implements online Flash Attention using PTX Tensor Core
MMA instructions. Inputs are fp16; the output accumulator is fp32.

### Tile geometry

```
Tensor  Shape in smem       MMA instruction
──────  ─────────────────   ──────────────────────────────────
Q tile  [Br=16, k=16] fp16  mma.sync.aligned.m16n8k16 (A operand)
K tile  [Bc=8,  k=16] fp16  mma.sync.aligned.m16n8k16 (B operand)
V tile  [Bc=8,  n=8]  fp16  mma.sync.aligned.m16n8k8  (B operand, ldmatrix.trans)
P tile  [Br=16, Bc=8] fp32  softmax numerators, packed fp16 for PV MMA
O tile  [Br=16, n=8]  fp32  mma.sync.aligned.m16n8k8  (C/D accumulator)
```

Tile constants (defined in `flash_attn.cu`):

| Constant        | Value              | Role                                |
|-----------------|--------------------|-------------------------------------|
| `kFlashWarps`   | 4                  | warps per CUDA block                |
| `kFlashBr`      | `kFlashWarps × 16` | Q rows per block (64)               |
| `kFlashBcDefault` | 128             | KV rows per smem block (tunable)    |
| `kFlashBcMin`   | 8                  | minimum KV block size               |

### Shared memory layout

```
smem_raw:
  [0             .. kFlashBr * D)            — smem_q  (Q tile, fp16)
  [kFlashBr * D  .. (kFlashBr + Bc) * D)    — smem_k  (K block, fp16)
  [(kFlashBr+Bc)*D .. (kFlashBr+2*Bc)*D)    — smem_v  (V block, fp16)

smem_bytes = (kFlashBr + 2 * kv_block_rows) * D * sizeof(fp16)

D=128, kv_block_rows=128:  (64 + 256) * 128 * 2 = 81,920 B ≈ 80 KB  ✓ (< 128 KB)
D=256, kv_block_rows=64:   (64 + 128) * 256 * 2 = 98,304 B ≈ 96 KB  ✓
```

### Multi-warp design (warp-per-Q-row-tile)

Each CUDA block covers `kFlashBr = 64` Q rows. The four warps partition those rows
with no inter-warp communication required:

```
warp 0 → rb = 0, 4, 8, ...   (rows  0–15,  64–79, ...)
warp 1 → rb = 1, 5, 9, ...   (rows 16–31,  80–95, ...)
warp 2 → rb = 2, 6, 10, ...  (rows 32–47,  96–111, ...)
warp 3 → rb = 3, 7, 11, ...  (rows 48–63, 112–127, ...)
```

Each Q row tile has its own `Mi[rb]`, `Li[rb]`, `Oi[rb,:]` accumulators in HBM.
Two warps never touch the same accumulator row, so no barriers are needed inside
the attention loop. All warps read the same K/V smem block (reads never conflict).

The grid shrinks by `kFlashWarps` in the Z dimension compared to a single-warp
design, keeping total warp count identical while giving the warp scheduler 4 warps
per SM to hide memory latency.

### Online softmax loop structure

```
flash_attn_kernel  (grid: B × H_q × ceil(S/kFlashBr)):
  load smem_q  [kFlashBr, D]

  for each kv_tile  (streams K/V from HBM into smem_k, smem_v):
    sync

    warp_flash_attn  (each warp: rb = warp_id, warp_id+4, ...):
      for each rb tile:
        c_frag = 0                               ← reset once per (rb, cb)
        for each k tile (depth):
          c_frag += Q[rb,k] @ K[cb,k].T * scale  ← mma.m16n8k16
        ← softmax fires here on full QKᵀ[rb,cb] ─
        row_max = reduce(c_frag)
        mi_new  = max(mi_old, row_max)
        P       = exp(c_frag - mi_new)
        li_new  = li_old * exp(mi_old - mi_new) + rowsum(P)
        write Mi, Li → HBM
        rescale = exp(mi_old - mi_new)
        for each co tile (D output columns):
          O[rb,co] = rescale * O[rb,co] + P @ V[cb,co]  ← mma.m16n8k8
    sync

  O /= Li   (normalize)
```

### Grid and block dimensions

```cpp
dim3 grid (B, H_q, (S + kFlashBr - 1) / kFlashBr);
dim3 block(kFlashWarps * 32);   // 128 threads = 4 warps
size_t smem = (kFlashBr + 2 * kv_block_rows) * D * sizeof(uint16_t);
```

### Known deviations from Flash Attention 2

| Deviation | Affects correctness | Notes |
|-----------|---------------------|-------|
| No causal mask | **Yes** — wrong output for decoder inference | Required before use in autoregressive generation |
| Q-len == KV-len | **Yes** — can't do decode (Q-len=1, KV-len=S) | `token_size` is shared; needs split into `q_len` / `kv_len` |
| O, Mi, Li in HBM each `(rb,cb)` | No — performance only | FA2 keeps these in registers across the full KV sweep; extra HBM traffic scales as `c_col_tiles × kv_tiles` |
| O output in fp32 | No | FA2 outputs fp16; fp32 doubles write bandwidth |
| No async K/V prefetch | No | FA2 uses `cp.async` to pipeline K/V loads with compute |

---

## Python Model

`kern_models/flash_attn.py` is a NumPy model of `kernels/flash_attn.cu` that
mirrors the exact tile geometry, loop structure, and online softmax. Use it to
verify algorithm changes before touching the CUDA code.

```bash
python kern_models/flash_attn.py
# D= 16  max_abs_err=0.000000  PASS
# D= 64  max_abs_err=0.000000  PASS
# D=128  max_abs_err=0.000000  PASS
```

Each function in the model corresponds 1-to-1 with a CUDA function and cites the
matching CUDA line number where the mapping is non-obvious.

---

## Benchmarking

### C++ GPU benchmark

`tests/bench_flash_attn.cu` measures GPU-side latency via CUDA Events (warmup=5,
iters=50). The same shapes are used in both the C++ and Python benchmarks.

```bash
# Build (from repo root)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target bench_flash_attn

# Run
./build/tests/bench_flash_attn
```

### PyTorch FA2 reference benchmark

`tests/bench_flash_attn2.py` benchmarks PyTorch `scaled_dot_product_attention`
with `SDPBackend.FLASH_ATTENTION` over identical shapes. GQA shapes (H_kv < H_q)
expand K/V via `repeat_interleave` to hit the Flash Attention path.

```bash
/home/sviraaj/projects/AI/.venv/bin/python tests/bench_flash_attn2.py
```

> **Note:** FA2 output is fp16; the C++ kernel writes fp32. Bandwidth numbers are
> not directly comparable — FA2's figure is lower because it writes half the bytes.

### Results — RTX 5070 (Blackwell sm_120, CUDA 12.9)

#### Prefill-like (B=1, H_q=32, H_kv=8, D=128)

| S    | warpforge (ms) | warpforge (TFLOPS) | FA2 (ms) | FA2 (TFLOPS) | Gap   |
|------|---------------:|-------------------:|---------:|-------------:|------:|
| 512  |   3.0          |  1.42              |  0.088   |  48.7        |  34×  |
| 1024 |  10.4          |  1.65              |  0.337   |  51.0        |  31×  |
| 2048 |  40.8          |  1.69              |  1.220   |  56.3        |  33×  |
| 4096 | 162.6          |  1.69              |  4.823   |  57.0        |  34×  |

#### Short-prefill / batched (D=128)

| Config                           | warpforge (ms) | TFLOPS | FA2 (ms) | FA2 TFLOPS | Gap  |
|----------------------------------|---------------:|-------:|---------:|-----------:|-----:|
| B=1 S=64  H_q=32 H_kv=8         |   0.090        |  0.74  |   0.007  |  10.1      |  14× |
| B=1 S=128 H_q=32 H_kv=8         |   0.289        |  0.93  |   0.010  |  25.8      |  28× |
| B=8 S=64  H_q=32 H_kv=8         |   0.542        |  0.99  |   0.033  |  16.3      |  16× |

#### Head dimension sweep (B=1, S=1024, H_q=32, H_kv=8)

| D   | warpforge (ms) | TFLOPS | FA2 (ms) | FA2 TFLOPS | Gap  |
|-----|---------------:|-------:|---------:|-----------:|-----:|
| 64  |   3.5          |  2.47  |  0.159   |  53.9      |  22× |
| 128 |  10.4          |  1.65  |  0.340   |  50.5      |  31× |
| 256 |  21.7          |  1.58  |  0.642   |  53.5      |  34× |

#### Performance history

| Change | S=1024 latency | TFLOPS | vs FA2 |
|--------|---------------:|-------:|-------:|
| Baseline (`kFlashWarps=1`) | 74 ms | 0.23 | 215× |
| `kFlashWarps=4` (warp-per-rb) | 10.4 ms | 1.65 | 31× |

The multi-warp change delivered a **7× speedup** by raising SM occupancy from 2%
to ~8% (4 warps × 1 block/SM with 80 KB smem).

#### Remaining gap to FA2

The residual 31× gap has three compounding sources:

| Source | Estimated cost |
|--------|---------------|
| O round-tripped to HBM every `(rb,cb)` tile | ~10–20× extra HBM writes vs FA2 |
| Mi/Li read+written to HBM every `(rb,cb)` tile | extra global accesses |
| fp32 O output (vs fp16) | 2× write bandwidth |
| No async K/V prefetch (`cp.async`) | stall on smem loads |

Next targets: keep O accumulator in registers across all `cb` tiles, and keep
Mi/Li in registers across the full KV sweep.

---

## GPU Profiling (Nsight Compute)

ncu profiling is integrated as CMake custom targets. All targets profile
`bench_flash_attn` with `--launch-skip 5` (skips warmup) and
`--launch-count 1` (profiles one kernel invocation).

`bench_flash_attn` is compiled with `-lineinfo` so reports can annotate
source lines in `kernels/flash_attn.cu`.

### Setup

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DNCU_PATH=/usr/local/cuda-12.9/bin
cmake --build build --target bench_flash_attn
```

All profiling commands must run from `build/`:

```bash
cd build
```

### Available targets

#### `ncu_stalls` — occupancy and stall breakdown (fast, stdout only)

```bash
sudo cmake --build . --target ncu_stalls
```

Key metrics reported:

| Metric | Scope | What it means |
|--------|-------|---------------|
| `sm__warps_active.avg.per_cycle_active` | per SM | Absolute active warp count — shows 4 with `kFlashWarps=4` and smem forcing 1 block/SM |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | per SM | Occupancy % (peak = 48–64 warps/SM on Blackwell) — ~8% with 4 warps |
| `smsp__warps_eligible.avg.per_cycle_active` | per SMSP | Eligible warps per SM sub-partition (SM has 4 SMSPs). Shows **1.0** with 4 warps/SM — this is correct, not a sign of 1 warp total |
| `stalled_long_scoreboard` (%) | per SMSP | Waiting on HBM loads — the dominant stall type |
| `stalled_wait` (%) | per SMSP | Waiting on `__syncthreads()` |

#### `ncu_bandwidth` — HBM, L2, smem byte counts

```bash
sudo cmake --build . --target ncu_bandwidth
```

Compare `dram__bytes_write` against the expected `Q_n * sizeof(fp32)` minimum.
The excess reveals O and Mi/Li HBM round-trips. With the current kernel the write
count scales as `c_col_tiles × kv_tiles × min_writes` instead of `min_writes`.

#### `ncu_tensorcore` — tensor core utilization

```bash
sudo cmake --build . --target ncu_tensorcore
```

`sm__pipe_tensor_op_hmma_cycles_active` is the HMMA pipe utilization percentage.
With the single-warp baseline this was <5%. With 4 warps expect ~15–30%.

#### `ncu_roofline` — roofline chart (saved to file)

```bash
sudo cmake --build . --target ncu_roofline
ncu-ui reports/ncu_roofline.ncu-rep
```

Shows where the kernel sits relative to the memory and compute rooflines on the
RTX 5070. The current kernel is memory-bandwidth-limited, well below the compute
roof due to the O HBM round-trips.

#### `ncu_full` — full section report (saved to file)

```bash
sudo cmake --build . --target ncu_full
ncu-ui reports/ncu_full.ncu-rep
```

Collects every built-in section including source-level annotation of
`kernels/flash_attn.cu`. Opens in the Nsight Compute GUI.

### Reports directory

Reports are saved to `build/reports/`. The directory is gitignored
(`*.ncu-rep`, `*.nsys-rep`).

---

## Repository Structure

```
warpforge/
├── include/
│   ├── tensor.h            # Tensor descriptor (shape, dtype, device ptr)
│   ├── model_config.h      # ModelConfig (Llama-8B defaults)
│   ├── kernels.h           # Core kernel declarations
│   ├── ops.h               # High-level op declarations (cuBLAS-backed)
│   ├── weights.h           # Weight structs and RuntimeHandles
│   ├── graph.h             # Computation graph node types
│   └── kv_cache.h          # Paged KV cache
│
├── src/
│   ├── kernels.cu          # ← Fill kernel implementations here
│   ├── ops.cu              # cuBLAS/cuBLASLt GEMM wrappers + op pipeline
│   ├── graph.cu            # Graph builder and printer
│   ├── weights.cu          # Weight allocation
│   ├── kv_cache.cu         # KV cache init / append
│   └── main.cu             # Inference loop entry point
│
├── kernels/
│   ├── flash_attn.cu       # Flash Attention — MMA tensor core implementation
│   ├── gemm_mma.cu         # m16n8k16 MMA GEMM (PTX inspection reference)
│   └── notes/
│       └── flash_attn_v1.txt   # Algorithm design notes and tiling strategy
│
├── kern_models/
│   └── flash_attn.py       # NumPy model of flash_attn.cu for algorithm verification
│
├── tests/
│   ├── test_utils.h        # CUDA_CHECK, tolerance helpers, RNG
│   ├── test_flash_attn.cu  # Flash Attention end-to-end correctness tests (18 cases)
│   ├── bench_flash_attn.cu # C++ GPU benchmark: latency / TFLOPS / bandwidth
│   ├── bench_flash_attn2.py # PyTorch FA2 (SDPA) reference benchmark
│   └── CMakeLists.txt
│
├── tools/
│   └── run_ptx.c           # CUDA Driver API PTX/cubin runner for kernel inspection
│
├── CMakeLists.txt
├── LICENSE
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| CMake | ≥ 3.20 |
| CUDA Toolkit | ≥ 12.0 (12.9 recommended) |
| GPU compute capability | sm_80+ (Ampere) / sm_120 (Blackwell RTX 5070) |
| cuBLAS / cuBLASLt | bundled with CUDA Toolkit |
| Python + PyTorch | for `bench_flash_attn2.py` and `kern_models/` only |
| ncu | bundled in CUDA Toolkit, for profiling targets |

The MMA instructions `mma.sync.m16n8k16` and `mma.sync.m16n8k8` require **sm_80 minimum**.
Primary development and tuning target is **RTX 5070** (Blackwell, sm_120).

---

## Building

```bash
# Configure for Blackwell (RTX 5070)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120

# Build everything
cmake --build build -j$(nproc)

# Build specific targets
cmake --build build --target bench_flash_attn
cmake --build build --target test_flash_attn
```

If ncu is not on `PATH`, point cmake to it at configure time:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DNCU_PATH=/usr/local/cuda-12.9/bin
```

---

## Running Tests

```bash
# Run all registered tests
cd build && ctest --output-on-failure

# Run Flash Attention correctness tests directly (18 shapes)
./build/tests/test_flash_attn
```

All 18 test cases cover small sequences, multi-tile KV loops, GQA (H_q > H_kv),
MQA (H_kv=1), batched inputs, varying head dimensions, and non-power-of-two
sequence lengths. All pass with max absolute error < 5×10⁻².

---

## Kernel Implementation Guide

When filling in a stub in `src/kernels.cu`:

1. The matching test in `tests/` provides a CPU reference and tolerance — this is
   the ground truth for correctness.
2. For `rmsnorm` and `swiglu`, a single block per token with a warp-level
   reduction is sufficient.
3. For `rope_apply`, each thread handles one `(cos, sin)` rotation pair.
4. For `attention`, see `kernels/flash_attn.cu` for the MMA tiling and online
   softmax pattern; see `kern_models/flash_attn.py` for the algorithm in pure NumPy.
5. Run `ctest` after each kernel before moving on.

---

## References

- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [NVIDIA PTX ISA — `mma` instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
- [NVIDIA Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)

---

## License

MIT — see [LICENSE](LICENSE).
