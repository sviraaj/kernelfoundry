#pragma once
#include "tensor.h"
#include "model_config.h"

// Empty kernels you will fill
void rmsnorm_empty(Tensor* out, const Tensor* x, const void* weight, const ModelConfig* cfg, cudaStream_t s);
void rope_apply_empty(Tensor* q, Tensor* k, int32_t pos0, const ModelConfig* cfg, cudaStream_t s);

// Attention path (you will implement sliding window + GQA)
void attention_empty(Tensor* ctx, const Tensor* q, const void* k_cache, const void* v_cache,
                     int32_t T_total, int32_t pos0, const ModelConfig* cfg, cudaStream_t s);

// SwiGLU path
void swiglu_empty(Tensor* z, const Tensor* u, const Tensor* v, cudaStream_t s);

// Sampling
int32_t sample_next_token_empty(const float* logits_host, int32_t vocab, float temperature);
