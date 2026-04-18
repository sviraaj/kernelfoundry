#include "kv_cache.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

int kv_cache_init_empty(KVPagedCache* kv, const ModelConfig* cfg, int32_t B, int32_t max_T) {
  memset(kv, 0, sizeof(*kv));
  kv->B = B;
  kv->max_T = max_T;
  kv->cur_T = 0;

  kv->k_cache = (void**)calloc(cfg->n_layers, sizeof(void*));
  kv->v_cache = (void**)calloc(cfg->n_layers, sizeof(void*));
  if (!kv->k_cache || !kv->v_cache) return -1;

  size_t KDh = (size_t)cfg->n_kv_heads * (size_t)cfg->head_dim; // 1024
  size_t elems_per_layer = (size_t)B * (size_t)max_T * KDh;
  size_t bytes = elems_per_layer * (size_t)cfg->bytes_per_elem;

  for (int l = 0; l < cfg->n_layers; l++) {
    cudaMalloc(&kv->k_cache[l], bytes);
    cudaMalloc(&kv->v_cache[l], bytes);
  }
  return 0;
}

void kv_cache_free(KVPagedCache* kv) {
  if (!kv) return;
  if (kv->k_cache) {
    // If you stored cfg inside kv, free properly; here we only free pointers array.
  }
}

int kv_cache_append_empty(KVPagedCache* kv, int layer, const void* K_new, const void* V_new,
                          int32_t B, int32_t S_new, const ModelConfig* cfg, cudaStream_t stream) {
  (void)B; (void)cfg; (void)stream;

  // Append K_new/V_new into cache at offset cur_T.
  // Implement proper layout: [B, T, K, Dh] or packed [B, T, KDh].
  // For now: placeholder.
  (void)kv; (void)layer; (void)K_new; (void)V_new; (void)S_new;
  return 0;
}
