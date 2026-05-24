// bench_flash_attn.cu — Throughput and bandwidth benchmark for flash_attn_kernel.
//
// Metrics reported per shape:
//   latency   (ms/iter) — GPU-side elapsed time via CUDA Events
//   TFLOPS              — 4·B·H_q·S²·D FLOPs (QKᵀ + PV, each 2·B·H_q·S²·D)
//   bandwidth (GB/s)    — (Q+K+V reads in fp16) + (O write in fp32)

#include "../kernels/flash_attn.cu"
#include "test_utils.h"

#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

static inline uint16_t f32_to_f16(float v) {
    __half h = __float2half(v);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

static inline size_t smem_bytes_for(int kv_block_rows, int D) {
    return static_cast<size_t>(kFlashBr + 4 * kv_block_rows) * D * sizeof(uint16_t);
}

static inline int select_bc(int D) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    int max_smem = 0;
    if (cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess)
        return -1;
    int bc = kFlashBcDefault;
    // We are doing max_smem such that the kernel need to double-buffer KV tiles.
    while (bc >= kFlashBcMin && smem_bytes_for(bc, D) > static_cast<size_t>(max_smem))
        bc >>= 1;
    return (bc >= kFlashBcMin) ? bc : -1;
}

static void launch_once(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    float* O, float* d_Mi, float* d_Li,
    int B, int q_len, int kv_len, int H_q, int H_kv, int D,
    int bc, size_t smem, cudaStream_t stream
) {
    const dim3 grid(B, H_q, (q_len + kFlashBr - 1) / kFlashBr);
    const dim3 block(kFlashWarps * 32);
#define LAUNCH(D_VAL) flash_attn_kernel<D_VAL><<<grid, block, smem, stream>>>(Q, K, V, O, d_Mi, d_Li, B, q_len, kv_len, H_q, H_kv, bc)
    if      (D ==  16) { LAUNCH( 16); }
    else if (D ==  64) { LAUNCH( 64); }
    else if (D == 128) { LAUNCH(128); }
    else if (D == 256) { LAUNCH(256); }
#undef LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// q_len=0 means q_len=kv_len (prefill / self-attention).
static void run_bench(int B, int kv_len, int H_q, int H_kv, int D,
                      int q_len = 0, int warmup = 5, int iters = 50) {
    if (q_len <= 0) q_len = kv_len;
    const bool decode = (q_len != kv_len);

    const int bc = select_bc(D);
    if (bc < 0) {
        printf("  B=%-2d S=%-4d H_q=%-2d H_kv=%-2d D=%-3d | smem too large, skip\n",
               B, kv_len, H_q, H_kv, D);
        return;
    }
    const size_t smem = smem_bytes_for(bc, D);

    if (smem > 49152) {
#define SET_ATTR(D_VAL) cudaFuncSetAttribute(flash_attn_kernel<D_VAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem))
        if      (D ==  16) { CUDA_CHECK(SET_ATTR( 16)); }
        else if (D ==  64) { CUDA_CHECK(SET_ATTR( 64)); }
        else if (D == 128) { CUDA_CHECK(SET_ATTR(128)); }
        else if (D == 256) { CUDA_CHECK(SET_ATTR(256)); }
#undef SET_ATTR
    }

    const size_t Q_n  = static_cast<size_t>(B) * H_q  * q_len  * D;
    const size_t KV_n = static_cast<size_t>(B) * H_kv * kv_len * D;
    const size_t mi_n = static_cast<size_t>(B) * H_q  * q_len;

    std::vector<float> h_Q_f(Q_n), h_K_f(KV_n), h_V_f(KV_n);
    fill_random_f32(h_Q_f, -1.0f, 1.0f, 42);
    fill_random_f32(h_K_f, -1.0f, 1.0f, 43);
    fill_random_f32(h_V_f, -1.0f, 1.0f, 44);

    std::vector<uint16_t> h_Q(Q_n), h_K(KV_n), h_V(KV_n);
    for (size_t i = 0; i < Q_n;  ++i) h_Q[i] = f32_to_f16(h_Q_f[i]);
    for (size_t i = 0; i < KV_n; ++i) h_K[i] = f32_to_f16(h_K_f[i]);
    for (size_t i = 0; i < KV_n; ++i) h_V[i] = f32_to_f16(h_V_f[i]);

    uint16_t *d_Q, *d_K, *d_V;
    float    *d_O, *d_Mi, *d_Li;
    CUDA_CHECK(cudaMalloc(&d_Q,  Q_n  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_K,  KV_n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_V,  KV_n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_O,  Q_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Mi, mi_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Li, mi_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), Q_n  * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), KV_n * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), KV_n * sizeof(uint16_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < warmup; ++i)
        launch_once(d_Q, d_K, d_V, d_O, d_Mi, d_Li, B, q_len, kv_len, H_q, H_kv, D, bc, smem, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i)
        launch_once(d_Q, d_K, d_V, d_O, d_Mi, d_Li, B, q_len, kv_len, H_q, H_kv, D, bc, smem, stream);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    const float ms = total_ms / static_cast<float>(iters);

    // FLOPs: QKᵀ + PV = 4·B·H_q·q_len·kv_len·D
    const double flops  = 4.0 * B * H_q * static_cast<double>(q_len) * kv_len * D;
    const double tflops = flops / (static_cast<double>(ms) * 1e9);

    // Bandwidth: Q(fp16) + K(fp16) + V(fp16) reads, O(fp32) write
    const double bytes  = 2.0 * Q_n + 2.0 * KV_n + 2.0 * KV_n + 4.0 * Q_n;
    const double bw_gbs = bytes / (static_cast<double>(ms) * 1e6);

    if (decode) {
        printf("  B=%-2d Q=1    KV=%-4d H_q=%-2d H_kv=%-2d D=%-3d | %7.3f ms | %7.3f TFLOPS | %7.1f GB/s\n",
               B, kv_len, H_q, H_kv, D, ms, tflops, bw_gbs);
    } else {
        printf("  B=%-2d S=%-4d       H_q=%-2d H_kv=%-2d D=%-3d | %7.3f ms | %7.3f TFLOPS | %7.1f GB/s\n",
               B, kv_len, H_q, H_kv, D, ms, tflops, bw_gbs);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_Mi); cudaFree(d_Li);
}

int main() {
    printf("=== Flash Attention Benchmark (warmup=5, iters=50) ===\n");
    printf("  %-4s %-5s %-5s %-6s %-4s | %-10s | %-14s | %-10s\n",
           "B", "S", "H_q", "H_kv", "D", "latency", "throughput", "bandwidth");

    printf("\n[decode: Q_len=1 (single new token), KV_len=full cache]\n");
    run_bench(1,  512, 32,  8, 128, /*q_len=*/1);
    run_bench(1, 1024, 32,  8, 128, /*q_len=*/1);
    run_bench(1, 2048, 32,  8, 128, /*q_len=*/1);
    run_bench(1, 4096, 32,  8, 128, /*q_len=*/1);
    run_bench(8,  512, 32,  8, 128, /*q_len=*/1);
    run_bench(8, 1024, 32,  8, 128, /*q_len=*/1);

    printf("\n[short-prefill: Q_len=KV_len, small S]\n");
    run_bench(1,   64, 32,  8, 128);
    run_bench(1,  128, 32,  8, 128);
    run_bench(1,  256, 32,  8, 128);
    run_bench(8,   64, 32,  8, 128);

    printf("\n[prefill: large S]\n");
    run_bench(1,  512, 32,  8, 128);
    run_bench(1, 1024, 32,  8, 128);
    run_bench(1, 2048, 32,  8, 128);
    run_bench(1, 4096, 32,  8, 128);

    printf("\n[head dim sweep: B=1 S=1024 H_q=32 H_kv=8]\n");
    run_bench(1, 1024, 32,  8,  64);
    run_bench(1, 1024, 32,  8, 128);
    run_bench(1, 1024, 32,  8, 256);

    printf("\n[MQA: H_kv=1]\n");
    run_bench(1,  512, 32,  1, 128);
    run_bench(1, 1024, 32,  1, 128);
    run_bench(1,  512, 32,  1, 128, /*q_len=*/1);
    run_bench(1, 1024, 32,  1, 128, /*q_len=*/1);

    return EXIT_SUCCESS;
}
