#pragma once
#include "tensor.h"
#include "weights.h"
#include "kv_cache.h"
#include "model_config.h"

// High-level ops (call cuBLAS/cuBLASLt under the hood)
int op_embed_empty(Tensor* X, const int32_t* tokens_dev, int32_t B, int32_t S,
                   const ModelWeights* W, const ModelConfig* cfg, RuntimeHandles* h);

int op_layer_prefill_empty(Tensor* Y, Tensor* X, int layer, int32_t B, int32_t S, int32_t pos0,
                           const ModelWeights* W, KVPagedCache* kv, const ModelConfig* cfg, RuntimeHandles* h);

int op_layer_decode_step_empty(Tensor* y, Tensor* x, int layer, int32_t B, int32_t pos,
                               const ModelWeights* W, KVPagedCache* kv, const ModelConfig* cfg, RuntimeHandles* h);

int op_lm_head_empty(Tensor* logits, const Tensor* X, const ModelWeights* W,
                     const ModelConfig* cfg, RuntimeHandles* h);
