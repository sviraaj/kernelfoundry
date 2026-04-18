#pragma once
#include "model_config.h"
#include <stdint.h>

typedef struct KVPagedCache {
  // You can implement paged attention (blocks/pages).
  // For now: a simple contiguous cache pointer per layer (device).
  void** k_cache; // per layer: [B, T, K, Dh]
  void** v_cache; // per layer: [B, T, K, Dh]

  int32_t max_T;
  int32_t B;
  int32_t cur_T;   // current filled length
} KVPagedCache;

// Create/free empty cache buffers
int kv_cache_init_empty(KVPagedCache* kv, const ModelConfig* cfg, int32_t B, int32_t max_T);
void kv_cache_free(KVPagedCache* kv);

// Append K,V for a step or a chunk (you fill implementation)
int kv_cache_append_empty(KVPagedCache* kv, int layer, const void* K_new, const void* V_new,
                          int32_t B, int32_t S_new, const ModelConfig* cfg, cudaStream_t stream);
