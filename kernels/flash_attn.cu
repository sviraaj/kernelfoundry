#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cfloat>
#include <cstdio>

// Holds up to 4 uint32 registers used as mma fragment operands.
struct WarpFrag {
    uint32_t reg[4];
};

// Load a 16x16 fp16 tile (A operand) from smem via ldmatrix.x4.
// row_stride: elements per row of the matrix in smem.
__device__ __forceinline__ WarpFrag smem_load_frag_a(const void* addr, int row_stride) {
    WarpFrag frag       = {{0u, 0u, 0u, 0u}};
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int row       = lane & 15;
    const int col       = ((lane >> 4) & 1) * 8;
    const char* ptr     = static_cast<const char*>(addr) + (row * row_stride + col) * sizeof(uint16_t);
    const uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    // TODO: tile count (x4) is hardcoded for 16x16; generalize based on actual tile size.
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]), "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(sptr)
    );
    return frag;
}

// Load a 16x8 fp16 tile (B operand) from smem via ldmatrix.x2.
// row_stride: elements per row of the matrix in smem.
__device__ __forceinline__ WarpFrag smem_load_frag_b(const void* addr, int row_stride) {
    WarpFrag frag       = {{0u, 0u, 0u, 0u}};
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int row       = lane & 7;
    const int col       = ((lane >> 3) & 1) * 8;
    const char* ptr     = static_cast<const char*>(addr) + (row * row_stride + col) * sizeof(uint16_t);
    const uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(sptr)
    );
    return frag;
}

// Load a transposed 8x8 fp16 tile (B operand for m16n8k8) from smem via ldmatrix.x1.trans.
// row_stride: elements per row of the source matrix in smem.
__device__ __forceinline__ uint32_t smem_load_frag_b8_trans(const void* addr, int row_stride) {
    uint32_t frag       = 0u;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int row       = lane & 7;
    const char* ptr     = static_cast<const char*>(addr) + row * row_stride * sizeof(uint16_t);
    const uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
        : "=r"(frag)
        : "r"(sptr)
    );
    return frag;
}

// Load Q's m16×k16 A-operand from swizzled smem via ldmatrix.x4.
// q_row_base = rb * 16, k_iter = loop variable k.
template <int kDHeadDim>
__device__ __forceinline__ WarpFrag smem_load_frag_a_swizzled(
    const uint16_t* Q_base, int q_row_base, int k_iter)
{
    WarpFrag frag = {{0u, 0u, 0u, 0u}};
    constexpr int kChunksPerRow = kDHeadDim / 8;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int row       = lane & 15;
    const int col_group = (lane >> 4) & 1;
    const int global_chunk = k_iter * 2 + col_group;
    const int abs_row   = q_row_base + row;
    const int phys_chunk = global_chunk ^ (abs_row & (kChunksPerRow - 1));
    const char* ptr = reinterpret_cast<const char*>(Q_base)
                    + (abs_row * kDHeadDim + phys_chunk * 8) * sizeof(uint16_t);
    const uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]), "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(sptr));
    return frag;
}

// Load K's k16×n8 B-operand from swizzled smem via ldmatrix.x2.
// k_row_base = cb * 8, k_iter = loop variable k.
template <int kDHeadDim>
__device__ __forceinline__ WarpFrag smem_load_frag_b_swizzled(
    const uint16_t* K_base, int k_row_base, int k_iter)
{
    WarpFrag frag = {{0u, 0u, 0u, 0u}};
    constexpr int kChunksPerRow = kDHeadDim / 8;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int row       = lane & 7;
    const int col_group = (lane >> 3) & 1;
    const int global_chunk = k_iter * 2 + col_group;
    const int abs_row   = k_row_base + row;
    const int phys_chunk = global_chunk ^ (abs_row & (kChunksPerRow - 1));
    const char* ptr = reinterpret_cast<const char*>(K_base)
                    + (abs_row * kDHeadDim + phys_chunk * 8) * sizeof(uint16_t);
    const uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(sptr));
    return frag;
}

// Load V's k8×n8 B-operand (transposed) from swizzled smem via ldmatrix.x1.trans.
// v_row_base = cb * 8, co = D-column block index.
template <int kDHeadDim>
__device__ __forceinline__ uint32_t smem_load_frag_b8_trans_swizzled(
    const uint16_t* V_base, int v_row_base, int co)
{
    uint32_t frag = 0u;
    constexpr int kChunksPerRow = kDHeadDim / 8;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = lane & 7;
    const int abs_row    = v_row_base + row;
    const int phys_chunk = co ^ (abs_row & (kChunksPerRow - 1));
    const char* ptr = reinterpret_cast<const char*>(V_base)
                    + (abs_row * kDHeadDim + phys_chunk * 8) * sizeof(uint16_t);
    const uint32_t sptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
        : "=r"(frag)
        : "r"(sptr));
    return frag;
}

// kDHeadDim is a compile-time constant, so:
//   - k_tiles and the k-loop bound are compile-time → fully unrolled
//   - o_frag co-loop bound is compile-time → fully unrolled, no local-memory spill
//   - scale is constant-folded by the compiler
template <int kDHeadDim>
__device__ void warp_flash_attn(
    const uint16_t* Q,
    const uint16_t* K,
    const uint16_t* V,   // full [kDHeadDim, Bc_total] col-major smem block
    int active_q_rows,
    int active_kv_cols,
    float& mi_u, float& mi_l,
    float& li_u, float& li_l,
    float (*o_frag)[4],
    int rb
) {
    if (Q == nullptr || K == nullptr) {
        return;
    }
    if (active_q_rows <= 0 || active_kv_cols <= 0) {
        return;
    }

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    constexpr int k_tiles     = kDHeadDim >> 4;
    const int c_col_tiles = (active_kv_cols + 7) >> 3;


    #ifdef DEBUG__
        const int c_row_tiles = (active_q_rows + 15) >> 4;
        printf("warp_flash_attn: block=(%d,%d,%d) threadIdx.x=%d DHeadDim=%d active_q_rows=%d active_kv_cols=%d c_row_tiles=%d c_col_tiles=%d k_tiles=%d Q=%p K=%p\n",
                blockIdx.x, blockIdx.y, blockIdx.z,
                threadIdx.x, kDHeadDim, active_q_rows, active_kv_cols, c_row_tiles, c_col_tiles, k_tiles, Q, K);
    #endif

    const int c_lane_col = (lane & 3) << 1;
    const int c_lane_row = lane >> 2;
    const int row_u = rb * 16 + c_lane_row;
    const int row_l = row_u + 8;
    const bool row_u_valid = row_u < active_q_rows;
    const bool row_l_valid = row_l < active_q_rows;

    for (int cb = 0; cb < c_col_tiles; ++cb) {
        const int tile_cols = active_kv_cols - (cb << 3);
        const bool col0_valid = c_lane_col < tile_cols;
        const bool col1_valid = (c_lane_col + 1) < tile_cols;

        float c_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int k = 0; k < k_tiles; ++k) {
            const WarpFrag q_frag = smem_load_frag_a_swizzled<kDHeadDim>(Q, rb * 16, k);
            const WarpFrag k_frag = smem_load_frag_b_swizzled<kDHeadDim>(K, cb * 8, k);

#ifdef DEBUG__
            // Each uint32 register holds two fp16 values: low=bits[15:0], high=bits[31:16]
            printf("warp_flash_attn tile: block=(%d,%d,%d) threadIdx.x=%d rb=%d cb=%d k=%d lane=%d A_frag=[%.4f %.4f | %.4f %.4f | %.4f %.4f | %.4f %.4f] B_frag=[%.4f %.4f | %.4f %.4f]\n",
                blockIdx.x, blockIdx.y, blockIdx.z,
                threadIdx.x, rb, cb, k, lane,
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[0] & 0xFFFF))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[0] >> 16))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[1] & 0xFFFF))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[1] >> 16))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[2] & 0xFFFF))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[2] >> 16))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[3] & 0xFFFF))),
                __half2float(__ushort_as_half((unsigned short)(q_frag.reg[3] >> 16))),
                __half2float(__ushort_as_half((unsigned short)(k_frag.reg[0] & 0xFFFF))),
                __half2float(__ushort_as_half((unsigned short)(k_frag.reg[0] >> 16))),
                __half2float(__ushort_as_half((unsigned short)(k_frag.reg[1] & 0xFFFF))),
                __half2float(__ushort_as_half((unsigned short)(k_frag.reg[1] >> 16)))
            );
#endif

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(c_frag[0]), "+f"(c_frag[1]), "+f"(c_frag[2]), "+f"(c_frag[3])
                : "r"(q_frag.reg[0]), "r"(q_frag.reg[1]), "r"(q_frag.reg[2]), "r"(q_frag.reg[3]),
                "r"(k_frag.reg[0]), "r"(k_frag.reg[1])
            );

#ifdef DEBUG__
            printf("warp_mma_qk_tile tile done: block=(%d,%d,%d) threadIdx.x=%d rb=%d cb=%d k=%d C_after=[%.4f %.4f %.4f %.4f]\n",
                    blockIdx.x, blockIdx.y, blockIdx.z,
                    threadIdx.x, rb, cb, k, c_frag[0], c_frag[1], c_frag[2], c_frag[3]);
#endif
        }
        // Flash Attention online softmax update — fires once per (rb, cb) after all
        // k-tiles have been accumulated into c_frag.
        // Step 1: scale QK^T by 1/sqrt(kDHeadDim) and compute per-row max.
        // MMA output layout (m16n8k16): lane t owns
        //   c_frag[0,1] -> row (t>>2),   cols (t&3)*2 and (t&3)*2+1
        //   c_frag[2,3] -> row (t>>2)+8, same cols
        // Lanes 4t..4t+3 share the same row; reduce across bits [1:0] via xor-shuffle.
        const float scale = 1.0f / sqrtf(static_cast<float>(kDHeadDim));
        c_frag[0] *= scale; c_frag[1] *= scale;
        c_frag[2] *= scale; c_frag[3] *= scale;

        if (!col0_valid) {
            c_frag[0] = -FLT_MAX;
            c_frag[2] = -FLT_MAX;
        }
        if (!col1_valid) {
            c_frag[1] = -FLT_MAX;
            c_frag[3] = -FLT_MAX;
        }
        if (!row_u_valid) {
            c_frag[0] = -FLT_MAX;
            c_frag[1] = -FLT_MAX;
        }
        if (!row_l_valid) {
            c_frag[2] = -FLT_MAX;
            c_frag[3] = -FLT_MAX;
        }

#ifdef DEBUG__
        printf("scores block=(%d,%d,%d) threadIdx.x=%d rb=%d cb=%d lane=%d scaled=[%.4f %.4f %.4f %.4f]\n",
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, rb, cb, lane, c_frag[0], c_frag[1], c_frag[2], c_frag[3]);
#endif

        float mnu = fmaxf(c_frag[0], c_frag[1]);   // local max, upper 8 rows
        float mnl = fmaxf(c_frag[2], c_frag[3]);   // local max, lower 8 rows

        mnu = fmaxf(mnu, __shfl_xor_sync(0xFFFFFFFF, mnu, 1));
        mnu = fmaxf(mnu, __shfl_xor_sync(0xFFFFFFFF, mnu, 2));
        mnl = fmaxf(mnl, __shfl_xor_sync(0xFFFFFFFF, mnl, 1));
        mnl = fmaxf(mnl, __shfl_xor_sync(0xFFFFFFFF, mnl, 2));

        // Capture old Mi before update (needed for O rescaling in Step 3).
        const float mi_old_u = row_u_valid ? mi_u : -FLT_MAX;
        const float mi_old_l = row_l_valid ? mi_l : -FLT_MAX;

        // Merge with running max from previous KV tiles.
        mnu = fmaxf(mnu, mi_old_u);
        mnl = fmaxf(mnl, mi_old_l);

        // Softmax numerators: e^(score - row_max).
        float Cs[4];
        Cs[0] = expf(c_frag[0] - mnu); Cs[1] = expf(c_frag[1] - mnu);
        Cs[2] = expf(c_frag[2] - mnl); Cs[3] = expf(c_frag[3] - mnl);

        // Step 2: update Mi and Li in registers.
        // Reduce per-row exp-sums across the 4 lanes sharing each row.
        float rsu = Cs[0] + Cs[1];
        float rsl = Cs[2] + Cs[3];
        rsu += __shfl_xor_sync(0xFFFFFFFF, rsu, 1);
        rsu += __shfl_xor_sync(0xFFFFFFFF, rsu, 2);
        rsl += __shfl_xor_sync(0xFFFFFFFF, rsl, 1);
        rsl += __shfl_xor_sync(0xFFFFFFFF, rsl, 2);

        // mnu/rsu are identical across all 4 row-sharing lanes (shuffle-reduced),
        // so each lane updates its own register copy consistently.
        if (row_u_valid) { li_u = li_u * expf(mi_old_u - mnu) + rsu; mi_u = mnu; }
        if (row_l_valid) { li_l = li_l * expf(mi_old_l - mnl) + rsl; mi_l = mnl; }

        // Step 3: update O via mma(P, V) using m16n8k8.
        //
        // A operand [m=16, k=8]: the softmax tile P = Cs, already in registers.
        //   m16n8k8 A layout: lane t holds A[t>>2, (t&3)*2 : +2] in one register.
        //   cvt.rn.f16x2 packs: dst[31:16]=hi_src, dst[15:0]=lo_src.
        uint32_t p_a[2];
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(p_a[0]) : "f"(Cs[1]), "f"(Cs[0]));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(p_a[1]) : "f"(Cs[3]), "f"(Cs[2]));

        // Rescale factors applied once per row, hoisted out of the co loop.
        const float rescale_u = expf(mi_old_u - mnu);
        const float rescale_l = expf(mi_old_l - mnl);

        // Iterate over 8-wide column tiles of O (and V).
        // Loop bound is compile-time → fully unrolled; o_frag[co] maps to named registers,
        // eliminating the local-memory (stack) spill seen with a runtime DHeadDim.
        #pragma unroll
        for (int co = 0; co < kDHeadDim / 8; ++co) {
            // B operand [k=8, n=8]: load the V tile through ldmatrix.trans so
            // row-major shared memory is consumed as the col-major fragment MMA expects.
            const uint32_t b0 = smem_load_frag_b8_trans_swizzled<kDHeadDim>(V, cb * 8, co);

#ifdef DEBUG__
            {
                printf("V block=(%d,%d,%d) threadIdx.x=%d rb=%d cb=%d co=%d lane=%d V_frag=[%.4f %.4f]\n",
                    blockIdx.x, blockIdx.y, blockIdx.z,
                    threadIdx.x, rb, cb, co, lane,
                    __half2float(__ushort_as_half((unsigned short)(b0 & 0xFFFF))),
                    __half2float(__ushort_as_half((unsigned short)(b0 >> 16))));
            }
#endif

            o_frag[co][0] *= rescale_u; o_frag[co][1] *= rescale_u;
            o_frag[co][2] *= rescale_l; o_frag[co][3] *= rescale_l;

            // O += P * V_tile  (m16 x n8 x k8, accumulates into o_frag).
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+f"(o_frag[co][0]), "+f"(o_frag[co][1]), "+f"(o_frag[co][2]), "+f"(o_frag[co][3])
                : "r"(p_a[0]), "r"(p_a[1]), "r"(b0)
            );

#ifdef DEBUG__
            printf("O block=(%d,%d,%d) threadIdx.x=%d rb=%d cb=%d co=%d lane=%d O_u=[%.4f %.4f] O_l=[%.4f %.4f]\n",
                blockIdx.x, blockIdx.y, blockIdx.z,
                threadIdx.x, rb, cb, co, lane,
                o_frag[co][0], o_frag[co][1], o_frag[co][2], o_frag[co][3]);
#endif
        } // end flash-attn update
    }
}

// Tile dimensions.
//   kFlashWarps = 4  — warps per block; each warp owns kFlashBr/kFlashWarps = 16 Q rows.
//   kFlashBr    = kFlashWarps*16 — Q rows per block; 4*16=64.
//   kv_block_rows (runtime) — KV rows streamed per iteration; all warps share the same K/V smem.
//   smem for D=128, kv_block_rows=128: (64+256)*128*2 = 81920 B ≈ 80 KB < 128 KB ✓
//
// Caller must set cudaFuncAttributeMaxDynamicSharedMemorySize before launch
// if the computed smem_bytes exceeds the default 48 KB.
static constexpr int kFlashWarps                    = 4;
static constexpr int kFlashBr                       = kFlashWarps * 16;
[[maybe_unused]] static constexpr int kFlashBcDefault = 128;
[[maybe_unused]] static constexpr int kFlashBcMin     = 8;

// flash_attn_kernel — top-level Flash Attention kernel.
//
// Tensor layout (all row-major): Q/K/V/O → [B, H, S, D]
//   Q  [B, H_q,  S, D] fp16
//   K  [B, H_kv, S, D] fp16
//   V  [B, H_kv, S, D] fp16
//   O  [B, H_q,  S, D] fp32  (output; written once per Q tile, after all KV tiles)
//   Mi [B, H_q,  S]    fp32  (running per-row max; initialized to -FLT_MAX inside)
//   Li [B, H_q,  S]    fp32  (running per-row sum-of-exp; initialized to 0 inside)
//
// Dynamic shared memory (pass smem_bytes at launch):
//   smem_bytes = (kFlashBr + 4*kv_block_rows) * kDHeadDim * sizeof(uint16_t)
//   For D=128, kv_block_rows=64:  (64+256)*128*2 = 81920 B ≈ 80 KB ✓
//   For D=256, kv_block_rows=32:  (64+128)*256*2 = 98304 B ≈ 96 KB ✓
//   Caller must call cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize,
//       smem_bytes) before the first launch when smem_bytes > 49152.
//
// Grid : dim3(batch_size, q_heads, ceil(S / kFlashBr))
// Block: dim3(kFlashWarps * 32)   →   128 threads when kFlashWarps = 4
//
// kDHeadDim is a compile-time template parameter. The launcher dispatches to the
// correct instantiation based on the runtime head dimension.
template <int kDHeadDim>
__global__ void flash_attn_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    float*          __restrict__ O,
    float*          __restrict__ Mi,
    float*          __restrict__ Li,
    int batch_size,
    int q_len,        // query sequence length  (= kv_len for prefill, = 1 for decode)
    int kv_len,       // key/value sequence length
    int q_heads,
    int kv_heads,
    int kv_block_rows // must be a positive multiple of 8; runtime-tunable for best performance
) {
    const int b      = static_cast<int>(blockIdx.x);
    const int h_q    = static_cast<int>(blockIdx.y);
    const int q_tile = static_cast<int>(blockIdx.z);

    if (b >= batch_size || h_q >= q_heads) return;

    // GQA: map each query head to its key/value head.
    const int h_kv = h_q / (q_heads / kv_heads);

    const int q_row_start = q_tile * kFlashBr;
    if (q_row_start >= q_len) return;
    const int q_rows = min(kFlashBr, q_len - q_row_start);

    // [B, H, S, D]: within a head, rows are contiguous in memory.
    const uint16_t* Q_block  = Q  + (b * q_heads  + h_q)  * q_len  * kDHeadDim
                                  + q_row_start * kDHeadDim;
    const uint16_t* K_head   = K  + (b * kv_heads + h_kv) * kv_len * kDHeadDim;
    const uint16_t* V_head   = V  + (b * kv_heads + h_kv) * kv_len * kDHeadDim;
    float*          O_block  = O  + (b * q_heads  + h_q)  * q_len  * kDHeadDim
                                  + q_row_start * kDHeadDim;
    float*          Mi_block = Mi + (b * q_heads  + h_q)  * q_len  + q_row_start;
    float*          Li_block = Li + (b * q_heads  + h_q)  * q_len  + q_row_start;

    // Carve dynamic shared memory (double-buffered K/V for cp.async pipelining):
    //   smem_q    [kFlashBr * kDHeadDim]       fp16  (Q tile, single buffer)
    //   smem_k[0] [kv_block_rows * kDHeadDim]  fp16  (K ping)
    //   smem_v[0] [kv_block_rows * kDHeadDim]  fp16  (V ping)
    //   smem_k[1] [kv_block_rows * kDHeadDim]  fp16  (K pong)
    //   smem_v[1] [kv_block_rows * kDHeadDim]  fp16  (V pong)
    // Total: (kFlashBr + 4 * kv_block_rows) * kDHeadDim * sizeof(fp16)
    extern __shared__ char smem_raw[];
    const int    kv_buf_elems  = kv_block_rows * kDHeadDim;
    uint16_t*    smem_q        = reinterpret_cast<uint16_t*>(smem_raw);
    uint16_t*    smem_k_buf[2] = {
        smem_q + kFlashBr * kDHeadDim,
        smem_q + kFlashBr * kDHeadDim + 2 * kv_buf_elems
    };
    uint16_t*    smem_v_buf[2] = {
        smem_k_buf[0] + kv_buf_elems,
        smem_k_buf[1] + kv_buf_elems
    };

    const int tid  = static_cast<int>(threadIdx.x);
    const int bdim = static_cast<int>(blockDim.x);
    const int warp_id = tid >> 5;


    // Load Q tile into smem with XOR swizzle (per-chunk uint4 writes).
    constexpr int kChunksPerRow = kDHeadDim / 8;
    for (int ci = tid; ci < kFlashBr * kChunksPerRow; ci += bdim) {
        const int row   = ci / kChunksPerRow;
        const int chunk = ci % kChunksPerRow;
        const int phys_chunk = chunk ^ (row & (kChunksPerRow - 1));
        uint4* dst = reinterpret_cast<uint4*>(smem_q) + row * kChunksPerRow + phys_chunk;
        if (row < q_rows) {
            *dst = reinterpret_cast<const uint4*>(Q_block)[row * kChunksPerRow + chunk];
        } else {
            *dst = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Stream KV tiles with cp.async double buffering.
    //
    // Pattern (software pipeline, depth 2):
    //   prologue : zero buf[0], async-copy tile 0 → buf[0], wait
    //   loop i   : async-copy tile i+1 → buf[1-cur]   (overlaps with compute below)
    //              warp_flash_attn on buf[cur]
    //              wait for tile i+1  (wait_async = cp.async.wait_group + __syncthreads)
    //              swap cur
    //
    // The zero-before-copy ensures out-of-bounds rows (edge tile) read as zero
    // inside warp_flash_attn, matching the previous conditional-store behaviour.
    const int kv_tiles = (kv_len + kv_block_rows - 1) / kv_block_rows;

    auto wait_async = [&]() {
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();
    };

    // Issue async load of tile t into double-buffer slot buf with XOR swizzle.
    // __syncthreads() inside guarantees all threads finish zeroing before the
    // async copy engine starts writing (prevents write-after-write on padding).
    auto async_load_kv = [&](int t, int buf) {
        const int ks = t * kv_block_rows;
        const int kr = min(kv_block_rows, kv_len - ks);
        constexpr int kChunksPerRow = kDHeadDim / 8;
        for (int i = tid; i < kv_buf_elems; i += bdim) {
            smem_k_buf[buf][i] = 0;
            smem_v_buf[buf][i] = 0;
        }
        __syncthreads();
        const int total_ops = kr * kChunksPerRow;
        for (int ci = tid; ci < total_ops; ci += bdim) {
            const int row   = ci / kChunksPerRow;
            const int chunk = ci % kChunksPerRow;
            const int phys_chunk = chunk ^ (row & (kChunksPerRow - 1));
            const uint16_t* src_k = K_head + (ks + row) * kDHeadDim + chunk * 8;
            const uint16_t* src_v = V_head + (ks + row) * kDHeadDim + chunk * 8;
            uint16_t* dst_k = smem_k_buf[buf] + row * kDHeadDim + phys_chunk * 8;
            uint16_t* dst_v = smem_v_buf[buf] + row * kDHeadDim + phys_chunk * 8;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_k))),
                   "l"(static_cast<const void*>(src_k)));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_v))),
                   "l"(static_cast<const void*>(src_v)));
        }
        asm volatile("cp.async.commit_group;\n" :::);
    };

    // Prologue: fill buf[0] with tile 0 and wait before entering the loop.
    async_load_kv(0, 0);
    wait_async();


    int cur = 0;
    const int c_row_tiles = (q_rows + 15) >> 4;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    // Each warp owns at most one rb tile (c_row_tiles <= kFlashWarps guaranteed).
    const int rb = (warp_id < c_row_tiles) ? warp_id : -1;

    // O, Mi, Li accumulators in registers — no HBM round-trip per KV tile.
    // kDHeadDim is compile-time, so this array size is known at compile time.
    // With full loop unrolling in warp_flash_attn, every o_frag[co] becomes a named
    // register group — no spill to the per-thread local memory depot.
    float o_frag[kDHeadDim / 8][4] = {};
    float mi_u_reg = -FLT_MAX, mi_l_reg = -FLT_MAX;
    float li_u_reg = 0.0f,     li_l_reg = 0.0f;

    const int c_lane_col = (lane & 3) << 1;
    const int c_lane_row = lane >> 2;
    const int row_u = (rb >= 0) ? rb * 16 + c_lane_row : 0;
    const int row_l = row_u + 8;
    const bool row_u_valid = (rb >= 0) && (row_u < q_rows);
    const bool row_l_valid = (rb >= 0) && (row_l < q_rows);

    // kv_tile loop stays block-level: all warps participate in every barrier and
    // async_load_kv call, preventing deadlock on partial Q blocks.
    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const int kv_rows = min(kv_block_rows, kv_len - kv_tile * kv_block_rows);

        if (kv_tile + 1 < kv_tiles)
            async_load_kv(kv_tile + 1, 1 - cur);

        if (rb >= 0)
            warp_flash_attn<kDHeadDim>(
                smem_q, smem_k_buf[cur], smem_v_buf[cur],
                q_rows, kv_rows,
                mi_u_reg, mi_l_reg, li_u_reg, li_l_reg, o_frag, rb
            );

        if (kv_tile + 1 < kv_tiles) {
            wait_async();
            cur ^= 1;
        }
    }
    __syncthreads();

    // Write O, Mi, Li from registers to HBM once, after sweeping all KV tiles.
    if (rb >= 0) {
        #pragma unroll
        for (int co = 0; co < kDHeadDim / 8; ++co) {
            const int o_u_offset = row_u * kDHeadDim + co * 8 + c_lane_col;
            const int o_l_offset = row_l * kDHeadDim + co * 8 + c_lane_col;
            if (row_u_valid) {
                O_block[o_u_offset]     = o_frag[co][0];
                O_block[o_u_offset + 1] = o_frag[co][1];
            }
            if (row_l_valid) {
                O_block[o_l_offset]     = o_frag[co][2];
                O_block[o_l_offset + 1] = o_frag[co][3];
            }
        }
        // Mi/Li: 4 lanes share each row; lane 0 of each group does the single store.
        if ((lane & 3) == 0) {
            if (row_u_valid) { Mi_block[row_u] = mi_u_reg; Li_block[row_u] = li_u_reg; }
            if (row_l_valid) { Mi_block[row_l] = mi_l_reg; Li_block[row_l] = li_l_reg; }
        }
    }
    __syncthreads();

    // Normalize: O[row, :] /= Li[row].
    for (int i = tid; i < q_rows * kDHeadDim; i += blockDim.x)
        O_block[i] /= Li_block[i / kDHeadDim];
}

// Explicit instantiations — one per supported head dimension.
// Add a new line here (and a case in the launcher) when supporting a new D value.
template __global__ void flash_attn_kernel<16>(
    const uint16_t*, const uint16_t*, const uint16_t*,
    float*, float*, float*, int, int, int, int, int, int);
template __global__ void flash_attn_kernel<64>(
    const uint16_t*, const uint16_t*, const uint16_t*,
    float*, float*, float*, int, int, int, int, int, int);
template __global__ void flash_attn_kernel<128>(
    const uint16_t*, const uint16_t*, const uint16_t*,
    float*, float*, float*, int, int, int, int, int, int);
template __global__ void flash_attn_kernel<256>(
    const uint16_t*, const uint16_t*, const uint16_t*,
    float*, float*, float*, int, int, int, int, int, int);
