#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

void rmsnorm_empty(Tensor* out, const Tensor* x, const void* weight, const ModelConfig* cfg, cudaStream_t s) {
  (void)out; (void)x; (void)weight; (void)cfg; (void)s;
  // TODO: kernel: per token RMS over D, scale by weight
}

void rope_apply_empty(Tensor* q, Tensor* k, int32_t pos0, const ModelConfig* cfg, cudaStream_t s) {
  (void)q; (void)k; (void)pos0; (void)cfg; (void)s;
  // TODO: kernel: apply RoPE to Q,K for positions [pos0..pos0+S-1]
}

void attention_empty(Tensor* ctx, const Tensor* q, const void* k_cache, const void* v_cache,
                     int32_t T_total, int32_t pos0, const ModelConfig* cfg, cudaStream_t s) {
  (void)ctx; (void)q; (void)k_cache; (void)v_cache; (void)T_total; (void)pos0; (void)cfg; (void)s;
  // TODO: kernel or cuBLAS-based attention:
  // - GQA mapping: kv_head = q_head / (H/K)
  // - causal + sliding window mask
  // - softmax + matmul with V
}

void swiglu_empty(Tensor* z, const Tensor* u, const Tensor* v, cudaStream_t s) {
  (void)z; (void)u; (void)v; (void)s;
  // TODO: kernel: z = silu(u) * v
}

int32_t sample_next_token_empty(const float* logits_host, int32_t vocab, float temperature) {
  (void)logits_host; (void)vocab; (void)temperature;
  // TODO: implement greedy/top-p/top-k on host or device.
  return 0;
}
