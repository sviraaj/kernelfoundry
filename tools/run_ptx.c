// run_ptx.c
//
// Runs our PTX Tensor Core kernel on GPU (CUDA Driver API), initializes A/B with random values,
// computes a CPU reference GEMM, compares GPU output vs CPU, and times ONLY the GPU kernel.
//
// Kernel expected in PTX:
//   .entry tc_mma_raw_fixed(.param .u64 A_ptr, .param .u64 B_ptr, .param .u64 Out_ptr)
//
// GPU kernel behavior assumed (from our PTX):
//   - blockDim.x MUST be 32 (one warp)
//   - gridDim.x = blocks
//   - Reads A: 16x16 half, row-major contiguous
//   - Reads B: 16x8  half, row-major contiguous
//   - Writes Out: per-lane 4 floats, out[(block*32 + lane)*4 + i] = c{i}
//
// CPU comparison mapping assumption (very important):
//   - mma m16n8k16 has 32 lanes each owning one 2x2 block of C(16x8)
//   - lane -> (rb, cb) where rb = lane % 8, cb = lane / 8
//   - (r0,c0) = (2*rb, 2*cb)
//   - c0,c1,c2,c3 correspond to:
//       c0 = C[r0+0][c0+0], c1 = C[r0+0][c0+1],
//       c2 = C[r0+1][c0+0], c3 = C[r0+1][c0+1]
//
// If your observed mapping differs (possible on some mma variants), tell me the mismatch pattern
// and we’ll fix the lane->(r,c) mapping table.
//
// Build:
//   gcc -O2 -Wall run_ptx.c -o run_ptx -lm -lcuda
//
// Run:
//   ./run_ptx --ptx kernels/gemm_16m8n16k.ptx --kernel tc_mma_raw_fixed --blocks 1 --seed 123
//

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define CUCHK(x) do {                                                   \
  CUresult _r = (x);                                                     \
  if (_r != CUDA_SUCCESS) {                                              \
    const char* name = NULL;                                             \
    const char* msg  = NULL;                                             \
    cuGetErrorName(_r, &name);                                           \
    cuGetErrorString(_r, &msg);                                          \
    fprintf(stderr, "CUDA Driver error %s: %s\n",                        \
            name ? name : "?", msg ? msg : "?");                         \
    exit(1);                                                             \
  }                                                                      \
} while(0)

static void* xmalloc(size_t n) {
  void* p = malloc(n);
  if (!p) { fprintf(stderr, "malloc failed\n"); exit(1); }
  return p;
}

static char* load_file_to_string(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror("fopen"); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) { fprintf(stderr, "Empty PTX file?\n"); exit(1); }
  char* buf = (char*)xmalloc((size_t)n + 1);
  size_t rd = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (rd != (size_t)n) { fprintf(stderr, "fread failed\n"); exit(1); }
  buf[n] = '\0';
  if (out_size) *out_size = (size_t)n;
  return buf;
}

// ---- float <-> half (IEEE 754 binary16) helpers ----

// float -> half bits (rounding: simple; good enough for test)
static uint16_t float_to_half_bits(float f) {
  union { float f; uint32_t u; } v;
  v.f = f;

  uint32_t sign = (v.u >> 31) & 1u;
  int32_t  exp  = (int32_t)((v.u >> 23) & 0xFFu) - 127;
  uint32_t mant = v.u & 0x7FFFFFu;

  if (((v.u >> 23) & 0xFFu) == 0xFFu) { // Inf/NaN
    uint16_t hExp = 0x1Fu;
    uint16_t hMant = (mant ? 0x200u : 0u); // qNaN
    return (uint16_t)((sign << 15) | (hExp << 10) | hMant);
  }

  if (exp > 15) { // overflow => Inf
    return (uint16_t)((sign << 15) | (0x1Fu << 10));
  }

  if (exp < -14) { // subnormal/underflow
    int32_t shift = (-14 - exp);
    if (shift > 24) {
      return (uint16_t)(sign << 15);
    }
    uint32_t mant24 = mant | 0x800000u;
    uint32_t sub = mant24 >> (shift + 13);
    uint32_t round_bit = (mant24 >> (shift + 12)) & 1u;
    sub += round_bit;
    return (uint16_t)((sign << 15) | (uint16_t)(sub & 0x3FFu));
  }

  uint16_t hExp = (uint16_t)(exp + 15);
  uint32_t mant_rounded = mant + 0x1000u;
  if (mant_rounded & 0x800000u) {
    mant_rounded = 0;
    hExp += 1;
    if (hExp >= 0x1Fu) {
      return (uint16_t)((sign << 15) | (0x1Fu << 10));
    }
  }
  uint16_t hMant = (uint16_t)((mant_rounded >> 13) & 0x3FFu);
  return (uint16_t)((sign << 15) | (hExp << 10) | hMant);
}

// half bits -> float
static float half_bits_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;

  union { uint32_t u; float f; } v;

  if (exp == 0) {
    if (mant == 0) {
      v.u = sign << 31;
      return v.f;
    }
    // subnormal
    // Normalize mantissa
    int e = -14;
    uint32_t m = mant;
    while ((m & 0x400u) == 0) { m <<= 1; e--; }
    m &= 0x3FFu;
    uint32_t fexp = (uint32_t)(e + 127);
    uint32_t fmant = m << 13;
    v.u = (sign << 31) | (fexp << 23) | fmant;
    return v.f;
  }

  if (exp == 0x1Fu) { // Inf/NaN
    v.u = (sign << 31) | (0xFFu << 23) | (mant ? 0x400000u : 0u);
    return v.f;
  }

  uint32_t fexp = (exp - 15 + 127);
  uint32_t fmant = mant << 13;
  v.u = (sign << 31) | (fexp << 23) | fmant;
  return v.f;
}

// Simple RNG float in [-1, 1]
static float frand_symmetric(void) {
  // rand() -> [0, RAND_MAX]
  float x = (float)rand() / (float)RAND_MAX; // [0,1]
  return 2.0f * x - 1.0f;
}

typedef enum {
  KERNEL_TYPE_PTX,
  KERNEL_TYPE_CUBIN,
} KernelType;

typedef struct {
  const char* kernel_path;
  const char* kernel_name;
  KernelType kernel_type;
  int blocks;
  int device_ordinal;
  unsigned int seed;

  // fixed for this kernel today
  int A_rows, A_cols; // 16x16
  int B_rows, B_cols; // 16x8
} RunConfig;

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage:\n"
    "  %s --kernel <file> --name <name> [--type ptx|cubin] [--blocks N] [--device D] [--seed S]\n"
    "\n"
    "Examples:\n"
    "  %s --kernel kernels/gemm_16m8n16k.ptx --name tc_mma_raw_fixed --type ptx --blocks 1 --seed 123\n"
    "  %s --kernel kernels/gemm_16m8n16k.cubin --name tc_mma_raw_fixed --type cubin --blocks 1 --seed 123\n",
    prog, prog, prog);
}

static KernelType parse_kernel_type(const char* s) {
  if (!strcmp(s, "ptx")) return KERNEL_TYPE_PTX;
  if (!strcmp(s, "cubin")) return KERNEL_TYPE_CUBIN;
  fprintf(stderr, "Bad kernel type: %s (use 'ptx' or 'cubin')\n", s);
  exit(1);
}

static KernelType detect_kernel_type(const char* path) {
  const char* dot = strrchr(path, '.');
  if (!dot) {
    fprintf(stderr, "Cannot determine kernel type from path: %s\n", path);
    exit(1);
  }
  if (!strcmp(dot, ".ptx")) return KERNEL_TYPE_PTX;
  if (!strcmp(dot, ".cubin")) return KERNEL_TYPE_CUBIN;
  if (!strcmp(dot, ".fatbin")) return KERNEL_TYPE_CUBIN;
  fprintf(stderr, "Unknown kernel extension: %s\n", dot);
  exit(1);
}

static RunConfig parse_args(int argc, char** argv) {
  RunConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.blocks = 1;
  cfg.device_ordinal = 0;
  cfg.seed = (unsigned int)time(NULL);
  cfg.kernel_type = -1;  // Not set yet

  cfg.A_rows = 16; cfg.A_cols = 16;
  cfg.B_rows = 16; cfg.B_cols = 8;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--kernel") && i + 1 < argc) {
      cfg.kernel_path = argv[++i];
    } else if (!strcmp(argv[i], "--name") && i + 1 < argc) {
      cfg.kernel_name = argv[++i];
    } else if (!strcmp(argv[i], "--type") && i + 1 < argc) {
      cfg.kernel_type = parse_kernel_type(argv[++i]);
    } else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) {
      cfg.blocks = parse_int(argv[++i]);
    } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
      cfg.device_ordinal = parse_int(argv[++i]);
    } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
      cfg.seed = parse_uint(argv[++i]);
    } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      usage(argv[0]); exit(0);
    } else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      usage(argv[0]); exit(1);
    }
  }

  if (!cfg.kernel_path || !cfg.kernel_name) {
    usage(argv[0]);
    exit(1);
  }

  // Auto-detect kernel type if not specified
  if (cfg.kernel_type == (KernelType)-1) {
    cfg.kernel_type = detect_kernel_type(cfg.kernel_path);
    printf("Auto-detected kernel type: %s\n", cfg.kernel_type == KERNEL_TYPE_PTX ? "PTX" : "CUBIN");
  }

  if (cfg.blocks <= 0) {
    fprintf(stderr, "--blocks must be > 0\n");
    exit(1);
  }
  return cfg;
}

static void print_device_info(CUdevice dev) {
  char name[256];
  int ccMajor = 0, ccMinor = 0;
  CUCHK(cuDeviceGetName(name, (int)sizeof(name), dev));
  CUCHK(cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CUCHK(cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  printf("Device: %s (CC %d.%d)\n", name, ccMajor, ccMinor);
}

typedef struct {
  CUcontext ctx;
  CUmodule  mod;
  CUfunction fun;
  CUstream  stream;
} KernelHandle;

static KernelHandle load_ptx_kernel(const RunConfig* cfg) {
  KernelHandle kh;
  memset(&kh, 0, sizeof(kh));

  CUCHK(cuInit(0));

  CUdevice dev;
  CUCHK(cuDeviceGet(&dev, cfg->device_ordinal));
  print_device_info(dev);

  CUCHK(cuCtxCreate(&kh.ctx, 0, dev));
  CUCHK(cuStreamCreate(&kh.stream, CU_STREAM_DEFAULT));

  size_t ptx_size = 0;
  char* ptx = load_file_to_string(cfg->kernel_path, &ptx_size);

  // JIT options: target from context, plus logs
  char infoLog[16384];
  char errorLog[16384];
  memset(infoLog, 0, sizeof(infoLog));
  memset(errorLog, 0, sizeof(errorLog));

  CUjit_option opts[6];
  void* optVals[6];

  unsigned int optLevel = 4;

  opts[0] = CU_JIT_OPTIMIZATION_LEVEL;
  optVals[0] = (void*)(uintptr_t)optLevel;

  opts[1] = CU_JIT_TARGET_FROM_CUCONTEXT;
  optVals[1] = (void*)(uintptr_t)1;

  opts[2] = CU_JIT_INFO_LOG_BUFFER;
  optVals[2] = (void*)infoLog;

  opts[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optVals[3] = (void*)(uintptr_t)sizeof(infoLog);

  opts[4] = CU_JIT_ERROR_LOG_BUFFER;
  optVals[4] = (void*)errorLog;

  opts[5] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optVals[5] = (void*)(uintptr_t)sizeof(errorLog);

  CUresult r = cuModuleLoadDataEx(&kh.mod, ptx, 6, opts, optVals);
  if (r != CUDA_SUCCESS) {
    const char* name = NULL; const char* msg = NULL;
    cuGetErrorName(r, &name); cuGetErrorString(r, &msg);
    fprintf(stderr, "cuModuleLoadDataEx failed: %s: %s\n", name ? name : "?", msg ? msg : "?");
    fprintf(stderr, "JIT error log:\n%s\n", errorLog);
    fprintf(stderr, "JIT info log:\n%s\n", infoLog);
    exit(1);
  }

  if (errorLog[0]) printf("JIT error log:\n%s\n", errorLog);
  if (infoLog[0])  printf("JIT info log:\n%s\n", infoLog);

  CUCHK(cuModuleGetFunction(&kh.fun, kh.mod, cfg->kernel_name));

  free(ptx);
  return kh;
}

static KernelHandle load_cubin_kernel(const RunConfig* cfg) {
  KernelHandle kh;
  memset(&kh, 0, sizeof(kh));

  CUCHK(cuInit(0));

  CUdevice dev;
  CUCHK(cuDeviceGet(&dev, cfg->device_ordinal));
  print_device_info(dev);

  CUCHK(cuCtxCreate(&kh.ctx, 0, dev));
  CUCHK(cuStreamCreate(&kh.stream, CU_STREAM_DEFAULT));

  size_t cubin_size = 0;
  char* cubin = load_file_to_string(cfg->kernel_path, &cubin_size);

  CUresult r = cuModuleLoadData(&kh.mod, cubin);
  if (r != CUDA_SUCCESS) {
    const char* name = NULL; const char* msg = NULL;
    cuGetErrorName(r, &name); cuGetErrorString(r, &msg);
    fprintf(stderr, "cuModuleLoadData failed: %s: %s\n", name ? name : "?", msg ? msg : "?");
    exit(1);
  }

  CUCHK(cuModuleGetFunction(&kh.fun, kh.mod, cfg->kernel_name));

  free(cubin);
  return kh;
}

static KernelHandle load_kernel(const RunConfig* cfg) {
  printf("Loading %s kernel from: %s\n",
         cfg->kernel_type == KERNEL_TYPE_PTX ? "PTX" : "CUBIN",
         cfg->kernel_path);

  if (cfg->kernel_type == KERNEL_TYPE_PTX) {
    return load_ptx_kernel(cfg);
  } else {
    return load_cubin_kernel(cfg);
  }
}

static void unload_kernel(KernelHandle* kh) {
  if (!kh) return;
  if (kh->mod) CUCHK(cuModuleUnload(kh->mod));
  if (kh->stream) CUCHK(cuStreamDestroy(kh->stream));
  if (kh->ctx) CUCHK(cuCtxDestroy(kh->ctx));
  memset(kh, 0, sizeof(*kh));
}

// CPU reference GEMM: C = A(16x16) * B(16x8), A/B are half bits row-major
static void cpu_gemm_16x16_16x8_f32(const uint16_t* A, const uint16_t* B, float* C) {
  // C is 16x8 row-major
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 16; k++) {
        float a = half_bits_to_float(A[i * 16 + k]);
        float b = half_bits_to_float(B[k * 8 + j]);
        acc += a * b;
      }
      C[i * 8 + j] = acc;
    }
  }
}

static inline int almost_equal(float a, float b, float atol, float rtol,
                               float* abs_err_out, float* rel_err_out) {
    float abs_err = fabsf(a - b);
    float denom = fmaxf(fabsf(b), 1e-6f);
    float rel_err = abs_err / denom;
    if (abs_err_out) *abs_err_out = abs_err;
    if (rel_err_out) *rel_err_out = rel_err;
    return (abs_err <= atol) || (rel_err <= rtol);
}

int compare_rowmajor_C16x8(const float* hOut, const float* Cref,
                           float atol, float rtol) {
    int mismatches = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    int max_i = -1;

    for (int i = 0; i < 16 * 8; i++) {
        float abs_err, rel_err;
        int ok = almost_equal(hOut[i], Cref[i], atol, rtol, &abs_err, &rel_err);
        if (!ok) mismatches++;

        if (abs_err > max_abs) { max_abs = abs_err; max_i = i; }
        if (rel_err > max_rel) max_rel = rel_err;
    }

    printf("Row-major compare C(16x8): mismatches=%d / %d\n",
           mismatches, 16 * 8);
    printf("max_abs_err=%.6g  max_rel_err=%.6g\n", max_abs, max_rel);

    if (max_i >= 0) {
        int r = max_i / 8;
        int c = max_i % 8;
        printf("Worst abs error at (r=%d,c=%d): GPU=% .7f  CPU=% .7f  abs=%.6g\n",
               r, c, hOut[max_i], Cref[max_i], max_abs);
    }
    return mismatches;
}

// ---------- Pretty Print Helpers ----------

void print_matrix_f32(const char* name, const float* M, int rows, int cols) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", M[i*cols + j]);
        }
        printf("\n");
    }
}

void print_matrix_half(const char* name, const uint16_t* M, int rows, int cols) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float f = half_bits_to_float(M[i*cols + j]);
            printf("%8.4f ", f);
        }
        printf("\n");
    }
}

#define A_MATRIX_ROWS 16
#define A_MATRIX_COLS 16
#define B_MATRIX_ROWS 16
#define B_MATRIX_COLS 8
#define OUT_MATRIX_ROWS 16
#define OUT_MATRIX_COLS 8

static void run_and_verify(const RunConfig* cfg, const KernelHandle* kh) {
  // fixed sizes for now
  const int A_elems = A_MATRIX_ROWS * A_MATRIX_COLS;
  const int B_elems = B_MATRIX_ROWS * B_MATRIX_COLS;

  const size_t A_bytes = (size_t)A_elems * sizeof(uint16_t);
  const size_t B_bytes = (size_t)B_elems * sizeof(uint16_t);

  // kernel output: blocks * 32 lanes * 4 floats
  const int out_floats = cfg->blocks * 32 * 4;
  const size_t Out_bytes = (size_t)out_floats * sizeof(float);

  // Host buffers
  uint16_t* hA = (uint16_t*)xmalloc(A_bytes);
  uint16_t* hB = (uint16_t*)xmalloc(B_bytes);
  float* hOut = (float*)xmalloc(Out_bytes);
  float* Cref = (float*)xmalloc(OUT_MATRIX_ROWS * OUT_MATRIX_COLS * sizeof(float));

  // Random init
  srand(cfg->seed);
  for (int i = 0; i < A_elems; i++) {
    float v = frand_symmetric();
    hA[i] = float_to_half_bits(v);
  }
  for (int i = 0; i < B_elems; i++) {
    float v = frand_symmetric();
    hB[i] = float_to_half_bits(v);
  }
  memset(hOut, 0, Out_bytes);

  // CPU ref (single tile)
  cpu_gemm_16x16_16x8_f32(hA, hB, Cref);

  // Device buffers
  CUdeviceptr dA = 0, dB = 0, dOut = 0;
  CUCHK(cuMemAlloc(&dA, A_bytes));
  CUCHK(cuMemAlloc(&dB, B_bytes));
  CUCHK(cuMemAlloc(&dOut, Out_bytes));

  CUCHK(cuMemcpyHtoDAsync(dA, hA, A_bytes, kh->stream));
  CUCHK(cuMemcpyHtoDAsync(dB, hB, B_bytes, kh->stream));
  CUCHK(cuMemsetD8Async(dOut, 0, Out_bytes, kh->stream));

  // Launch config: EXACTLY one warp per block
  int threads = 32;
  int blocks  = cfg->blocks;

  void* args[] = { (void*)&dA, (void*)&dB, (void*)&dOut };

  // Timing events (kernel-only timing)
  CUevent evStart, evStop;
  CUCHK(cuEventCreate(&evStart, CU_EVENT_DEFAULT));
  CUCHK(cuEventCreate(&evStop,  CU_EVENT_DEFAULT));

  // Ensure H2D is done before timing kernel
  CUCHK(cuStreamSynchronize(kh->stream));

  CUCHK(cuEventRecord(evStart, kh->stream));
  CUCHK(cuLaunchKernel(
    kh->fun,
    blocks, 1, 1,
    threads, 1, 1,
    0,
    kh->stream,
    args,
    NULL
  ));
  CUCHK(cuEventRecord(evStop, kh->stream));
  CUCHK(cuEventSynchronize(evStop));

  float ms = 0.0f;
  CUCHK(cuEventElapsedTime(&ms, evStart, evStop));

  // Copy back
  CUCHK(cuMemcpyDtoHAsync(hOut, dOut, Out_bytes, kh->stream));
  CUCHK(cuStreamSynchronize(kh->stream));

  // Verify: compare block 0’s 32 lanes to CPU ref mapping
  // (For blocks>1, inputs are the same in our PTX, so output repeats per block.)
  //const float atol = 1e-2f;  // tensor cores + half inputs can have small differences
  //const float rtol = 1e-2f;

  // ---------- Print everything ----------
  print_matrix_half("Matrix A (host)", hA, 16, 16);
  print_matrix_half("Matrix B (host)", hB, 16, 8);

  print_matrix_f32("C from CPU reference", Cref, 16, 8);
  print_matrix_f32("C from GPU (reconstructed)", hOut, 16, 8);

  int mism = compare_rowmajor_C16x8(hOut, Cref, /*atol=*/1e-2f, /*rtol=*/1e-2f);
  if (mism) printf("FAIL\n"); else printf("PASS\n");

  // Cleanup
  CUCHK(cuEventDestroy(evStart));
  CUCHK(cuEventDestroy(evStop));

  CUCHK(cuMemFree(dA));
  CUCHK(cuMemFree(dB));
  CUCHK(cuMemFree(dOut));

  free(hA);
  free(hB);
  free(hOut);
  free(Cref);
}

int main(int argc, char** argv) {
  RunConfig cfg = parse_args(argc, argv);

  KernelHandle kh = load_kernel(&cfg);
  run_and_verify(&cfg, &kh);
  unload_kernel(&kh);
  return 0;
}
