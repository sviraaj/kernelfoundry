#include "ops.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// NOTE: For GEMMs you will likely use:
// - cublasGemmEx for straightforward
// - cublasLtMatmul for fused/better perf
// Here: placeholders only.

static int gemm_empty(RuntimeHandles* h,
                      const Tensor* A, const void* W, // A:[B*S, in], W:[in, out]
                      Tensor* C) {                    // C:[B*S, out]
  (void)h; (void)A; (void)W; (void)C;
  // TODO: cublasLtMatmul / cublasGemmEx
  return 0;
}

int op_embed_empty(Tensor* X, const int32_t* tokens_dev, int32_t B, int32_t S,
                   const ModelWeights* W, const ModelConfig* cfg, RuntimeHandles* h) {
  (void)tokens_dev; (void)h;
  // X: [B, S, D]
  // TODO: embedding gather kernel from W->wte
  X->ndims = 3; X->shape[0]=B; X->shape[1]=S; X->shape[2]=cfg->d_model;
  return 0;
}

int op_layer_prefill_empty(Tensor* Y, Tensor* X, int layer, int32_t B, int32_t S, int32_t pos0,
                           const ModelWeights* W, KVPagedCache* kv, const ModelConfig* cfg, RuntimeHandles* h) {
  (void)Y; (void)B; (void)S; (void)pos0; (void)kv;

  const LayerWeights* L = &W->layers[layer];

  // --- Attention sub-block ---
  // A = rmsnorm(X)
  Tensor A = *X; // alias shape
  rmsnorm_empty(&A, X, L->rms1_weight, cfg, h->stream);

  // Q_lin = A @ Wq, K_lin = A @ Wk, V_lin = A @ Wv
  Tensor Qlin = {}; Tensor Klin = {}; Tensor Vlin = {};
  // TODO: allocate Qlin/Klin/Vlin buffers (device), set shapes:
  // Qlin: [B,S,D], Klin/Vlin: [B,S,K*Dh]
  gemm_empty(h, &A, L->wq, &Qlin);
  gemm_empty(h, &A, L->wk, &Klin);
  gemm_empty(h, &A, L->wv, &Vlin);

  // reshape to Q:[B,S,H,Dh], K/V:[B,S,K,Dh] (logical)
  Tensor Q = {}; Tensor Kt = {}; Tensor Vt = {};
  // TODO: view/reshape only (no copy) if you store packed appropriately

  // RoPE(Q,K)
  rope_apply_empty(&Q, &Kt, pos0, cfg, h->stream);

  // KV cache append
  // kv_cache_append_empty(kv, layer, Kt.data, Vt.data, B, S, cfg, h->stream);

  // Attention -> ctx:[B,S,H,Dh] then flatten -> [B,S,D]
  Tensor ctx = {}; // TODO allocate
  attention_empty(&ctx, &Q, kv->k_cache[layer], kv->v_cache[layer], kv->cur_T + S, pos0, cfg, h->stream);

  Tensor Aout = {}; // [B,S,D]
  gemm_empty(h, &ctx, L->wo, &Aout);

  // Residual: X = X + Aout
  // TODO: add kernel

  // --- MLP sub-block ---
  Tensor M = *X;
  rmsnorm_empty(&M, X, L->rms2_weight, cfg, h->stream);

  Tensor U = {}; Tensor V = {}; Tensor Z = {}; Tensor Mout = {};
  // U = M @ W1, V = M @ W3
  gemm_empty(h, &M, L->w1, &U);
  gemm_empty(h, &M, L->w3, &V);
  // Z = silu(U) * V
  swiglu_empty(&Z, &U, &V, h->stream);
  // Mout = Z @ W2
  gemm_empty(h, &Z, L->w2, &Mout);

  // Residual: Y = X + Mout
  // TODO: add kernel and set Y

  return 0;
}

int op_layer_decode_step_empty(Tensor* y, Tensor* x, int layer, int32_t B, int32_t pos,
                               const ModelWeights* W, KVPagedCache* kv, const ModelConfig* cfg, RuntimeHandles* h) {
  // Same as prefill but with S=1 and pos0=pos, and attention uses cache length pos+1 (sliding window)
  return op_layer_prefill_empty(y, x, layer, B, /*S=*/1, /*pos0=*/pos, W, kv, cfg, h);
}

int op_lm_head_empty(Tensor* logits, const Tensor* X, const ModelWeights* W,
                     const ModelConfig* cfg, RuntimeHandles* h) {
  (void)h;
  // logits: [B,S,V]
  // TODO: GEMM X:[B*S,D] @ lm_head:[D,V]
  (void)logits; (void)X; (void)W; (void)cfg;
  return 0;
}
