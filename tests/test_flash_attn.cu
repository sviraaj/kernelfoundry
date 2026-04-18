//
// test_flash_attn.cu — End-to-end correctness tests for flash_attn_kernel.
//
// Compares GPU output (fp16 Q/K/V → fp32 O via Flash Attention) against a
// CPU reference that computes standard scaled dot-product attention in fp32.
//
// Test cases cover: small sequences, multi-tile KV loops, GQA (H_q > H_kv),
// batched inputs, and varying head dimensions.

#include "../kernels/flash_attn.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "test_utils.h"

static constexpr float kTol = 5e-2f;

static inline size_t flash_smem_bytes_for(int kv_block_rows, int D) {
    return static_cast<size_t>(kFlashBr + 2 * kv_block_rows) * D * sizeof(uint16_t);
}

static inline int select_flash_bc(int D) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;

    int max_smem_optin = 0;
    err = cudaDeviceGetAttribute(
        &max_smem_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device
    );
    if (err != cudaSuccess) return -1;

    int kv_block_rows = kFlashBcDefault;
    while (kv_block_rows >= kFlashBcMin &&
           flash_smem_bytes_for(kv_block_rows, D) > static_cast<size_t>(max_smem_optin)) {
        kv_block_rows >>= 1;
    }

    return (kv_block_rows >= kFlashBcMin) ? kv_block_rows : -1;
}

// ---- Host-side launcher ---------------------------------------------------

inline cudaError_t launch_flash_attn(
    const uint16_t* Q,   // [B, H_q,  S, D] fp16, device
    const uint16_t* K,   // [B, H_kv, S, D] fp16, device
    const uint16_t* V,   // [B, H_kv, S, D] fp16, device
    float*          O,   // [B, H_q,  S, D] fp32, device (output)
    int B, int S, int H_q, int H_kv, int D,
    cudaStream_t stream = 0
) {
    const size_t mi_li_elems = static_cast<size_t>(B) * H_q * S;
    float *d_Mi = nullptr, *d_Li = nullptr;
    cudaError_t err;
    const int kv_block_rows = select_flash_bc(D);

    if (kv_block_rows < 0) return cudaErrorInvalidValue;

    err = cudaMalloc(&d_Mi, mi_li_elems * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_Li, mi_li_elems * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_Mi); return err; }

    const size_t smem_bytes = flash_smem_bytes_for(kv_block_rows, D);

    if (smem_bytes > 49152) {
        err = cudaFuncSetAttribute(
            flash_attn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (err != cudaSuccess) { cudaFree(d_Mi); cudaFree(d_Li); return err; }
    }

    const dim3 grid(B, H_q, (S + kFlashBr - 1) / kFlashBr);
    const dim3 block(kFlashWarps * 32);

    std::cout << "    launch flash_attn_kernel"
              << " grid=(" << grid.x << ", " << grid.y << ", " << grid.z << ")"
              << " block=(" << block.x << ", " << block.y << ", " << block.z << ")"
              << " smem_bytes=" << smem_bytes
              << " kv_block_rows=" << kv_block_rows
              << " params={B=" << B
              << ", S=" << S
              << ", H_q=" << H_q
              << ", H_kv=" << H_kv
              << ", D=" << D
              << ", Q=" << static_cast<const void*>(Q)
              << ", K=" << static_cast<const void*>(K)
              << ", V=" << static_cast<const void*>(V)
              << ", O=" << static_cast<void*>(O)
              << ", Mi=" << static_cast<void*>(d_Mi)
              << ", Li=" << static_cast<void*>(d_Li)
              << "}\n";

    flash_attn_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, d_Mi, d_Li, B, S, S, H_q, H_kv, D, kv_block_rows
    );

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        // Keep the per-launch softmax scratch buffers alive until the kernel
        // finishes; freeing them immediately after launch is a device UAF.
        err = cudaStreamSynchronize(stream);
    }

    cudaFree(d_Mi);
    cudaFree(d_Li);
    return err;
}

// ---- CPU reference --------------------------------------------------------
// Standard scaled dot-product attention in fp32.
// Q/K/V layout: [B, H, S, D]  (h_kv = h_q / (H_q/H_kv) for GQA)

static void cpu_attn(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>&       O,
    std::vector<float>*       C,
    int B, int S, int H_q, int H_kv, int D
) {
    const float scale     = 1.0f / std::sqrt(static_cast<float>(D));
    const int   gqa_ratio = H_q / H_kv;
    O.assign(static_cast<size_t>(B) * H_q * S * D, 0.0f);
    if (C != nullptr) {
        C->assign(static_cast<size_t>(B) * H_q * S * S, 0.0f);
    }

    std::vector<float> scores(S);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H_q; ++h) {
            const int h_kv = h / gqa_ratio;
            for (int i = 0; i < S; ++i) {
                const float* q_row = Q.data() + ((b * H_q  + h)    * S + i) * D;

                // Compute raw scores S[i, j] = Q[i] · K[j] * scale.
                for (int j = 0; j < S; ++j) {
                    const float* k_row = K.data() + ((b * H_kv + h_kv) * S + j) * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
                    scores[j] = dot * scale;
                    if (C != nullptr) {
                        const size_t c_idx = static_cast<size_t>(((b * H_q + h) * S + i) * S + j);
                        (*C)[c_idx] = scores[j];
                    }
                }

                // Softmax.
                float maxval = *std::max_element(scores.begin(), scores.end());
                float sumexp = 0.0f;
                for (int j = 0; j < S; ++j) { scores[j] = std::exp(scores[j] - maxval); sumexp += scores[j]; }
                for (int j = 0; j < S; ++j) scores[j] /= sumexp;

                // O[i] = sum_j p[j] * V[j].
                float* o_row = O.data() + ((b * H_q + h) * S + i) * D;
                for (int j = 0; j < S; ++j) {
                    const float* v_row = V.data() + ((b * H_kv + h_kv) * S + j) * D;
                    for (int d = 0; d < D; ++d) o_row[d] += scores[j] * v_row[d];
                }
            }
        }
    }
}

// ---- fp32 → fp16 conversion helper ----------------------------------------

static inline uint16_t f32_to_f16(float v) {
    __half h = __float2half(v);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

#ifdef DEBUG
static void print_tensor_bhsd(
    const char* name,
    const std::vector<float>& data,
    int B, int H, int S, int D
) {
    std::cout << name << " shape=[" << B << ", " << H << ", " << S << ", " << D << "]\n";
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::cout << "  " << name << "[b=" << b << ", h=" << h << "]\n";
            for (int s = 0; s < S; ++s) {
                std::cout << "    [";
                for (int d = 0; d < D; ++d) {
                    if (d > 0) std::cout << ", ";
                    const size_t idx = static_cast<size_t>(((b * H + h) * S + s) * D + d);
                    std::cout << data[idx];
                }
                std::cout << "]\n";
            }
        }
    }
}

static void print_tensor_bhss(
    const char* name,
    const std::vector<float>& data,
    int B, int H, int S
) {
    std::cout << name << " shape=[" << B << ", " << H << ", " << S << ", " << S << "]\n";
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::cout << "  " << name << "[b=" << b << ", h=" << h << "]\n";
            for (int i = 0; i < S; ++i) {
                std::cout << "    [";
                for (int j = 0; j < S; ++j) {
                    if (j > 0) std::cout << ", ";
                    const size_t idx = static_cast<size_t>(((b * H + h) * S + i) * S + j);
                    std::cout << data[idx];
                }
                std::cout << "]\n";
            }
        }
    }
}
#endif

// ---- Single test case ------------------------------------------------------

static bool run_case(int B, int S, int H_q, int H_kv, int D, uint32_t seed) {
    std::cout << "  B=" << B << " S=" << S
              << " H_q=" << H_q << " H_kv=" << H_kv
              << " D=" << D << " ... " << std::flush;

    const size_t Q_n  = static_cast<size_t>(B) * H_q  * S * D;
    const size_t KV_n = static_cast<size_t>(B) * H_kv * S * D;
    const size_t O_n  = Q_n;

    std::vector<float> h_Q(Q_n), h_K(KV_n), h_V(KV_n);
    fill_random_f32(h_Q, -1.0f, 1.0f, seed);
    fill_random_f32(h_K, -1.0f, 1.0f, seed + 1);
    fill_random_f32(h_V, -1.0f, 1.0f, seed + 2);

    // CPU reference (fp32).
    std::vector<float> ref_O;
    std::vector<float> ref_C;
    cpu_attn(h_Q, h_K, h_V, ref_O, &ref_C, B, S, H_q, H_kv, D);

    // Convert Q/K/V to fp16 for GPU.
    std::vector<uint16_t> h_Q16(Q_n), h_K16(KV_n), h_V16(KV_n);
    for (size_t i = 0; i < Q_n;  ++i) h_Q16[i] = f32_to_f16(h_Q[i]);
    for (size_t i = 0; i < KV_n; ++i) h_K16[i] = f32_to_f16(h_K[i]);
    for (size_t i = 0; i < KV_n; ++i) h_V16[i] = f32_to_f16(h_V[i]);

    uint16_t *d_Q, *d_K, *d_V;
    float    *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, Q_n  * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_K, KV_n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_V, KV_n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_O, O_n  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q16.data(), Q_n  * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K16.data(), KV_n * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V16.data(), KV_n * sizeof(uint16_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(launch_flash_attn(d_Q, d_K, d_V, d_O, B, S, H_q, H_kv, D));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got_O;
    download_f32(got_O, d_O, static_cast<int>(O_n));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    const float err = max_abs_error(ref_O, got_O);
    if (err > kTol) {
        std::cout << "FAIL (max_abs_err=" << err << ", tol=" << kTol << ")\n";
#ifdef DEBUG
        print_tensor_bhsd("Q", h_Q, B, H_q, S, D);
        print_tensor_bhsd("K", h_K, B, H_kv, S, D);
        print_tensor_bhsd("V", h_V, B, H_kv, S, D);
        print_tensor_bhss("C", ref_C, B, H_q, S);
        print_tensor_bhsd("expected_O", ref_O, B, H_q, S, D);
        print_tensor_bhsd("got_O", got_O, B, H_q, S, D);
#endif
        return false;
    }
    std::cout << "PASS (max_abs_err=" << err << ")\n";
    return true;
}

// ---- Main ------------------------------------------------------------------

int main() {
    std::cout << "=== Flash Attention end-to-end tests ===\n";
    int failed = 0;

    // Small sequence — single KV tile, single head.
    if (!run_case(1,  16,  1, 1,  16, 100)) ++failed;
    if (!run_case(1,  16,  1, 1,  64, 101)) ++failed;
    if (!run_case(1,  64,  1, 1, 128, 102)) ++failed;
    if (!run_case(1,  17,  1, 1, 128, 103)) ++failed;  // Partial Q tile (Br=16).

    // Multi-tile KV loop and ragged KV tails (Bc = 128).
    if (!run_case(1, 129,  1, 1, 128, 202)) ++failed;  // One full KV tile + tail.
    if (!run_case(1, 255,  1, 1, 128, 203)) ++failed;  // Near 2*Bc with ragged tail.
    if (!run_case(1, 256,  1, 1,  64, 200)) ++failed;
    if (!run_case(1, 256,  1, 1, 128, 201)) ++failed;

    // GQA/MQA shapes seen in modern models.
    if (!run_case(1,  64,  4, 2,  64, 300)) ++failed;
    if (!run_case(1,  64,  8, 2,  64, 301)) ++failed;
    if (!run_case(1,  64, 32, 8, 128, 302)) ++failed;  // Llama/Mistral-style GQA.
    if (!run_case(1,  64, 40, 8, 128, 303)) ++failed;  // Non-power-of-two GQA ratio.
    if (!run_case(1,  64,  8, 1, 128, 304)) ++failed;  // MQA extreme.
    if (!run_case(1,  16,  1, 1, 256, 305)) ++failed;  // Larger D with reduced KV tile.

    // Batched and mixed-shape stress.
    if (!run_case(2,  64,  4, 4,  64, 400)) ++failed;
    if (!run_case(2, 128,  4, 2, 128, 401)) ++failed;
    if (!run_case(3,  33,  4, 4, 128, 402)) ++failed;  // Ragged Q tiles across batches.
    if (!run_case(2, 129, 32, 8, 128, 403)) ++failed;  // Batch + GQA + KV tail.

    if (failed) {
        std::cout << failed << " test(s) FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Flash Attention tests PASSED.\n";
    return EXIT_SUCCESS;
}
