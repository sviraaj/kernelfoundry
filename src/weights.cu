#include "weights.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

static void* dev_alloc(size_t bytes) {
  void* p = NULL;
  cudaMalloc(&p, bytes);
  return p;
}

int load_weights_empty(const char* path, const ModelConfig* cfg, ModelWeights* W, RuntimeHandles* h) {
  (void)path; (void)h;
  memset(W, 0, sizeof(*W));

  // Allocate placeholder device buffers (NO actual loading).
  // Replace with mmap/ifstream + cudaMemcpyAsync, or unified memory, etc.

  size_t wte_bytes = (size_t)cfg->vocab_size * (size_t)cfg->d_model * cfg->bytes_per_elem;
  W->wte = dev_alloc(wte_bytes);

  W->rms_final = dev_alloc((size_t)cfg->d_model * cfg->bytes_per_elem);

  // lm_head optional (tie to wte if you want)
  W->lm_head = NULL;

  W->layers = (LayerWeights*)calloc(cfg->n_layers, sizeof(LayerWeights));
  for (int l = 0; l < cfg->n_layers; l++) {
    LayerWeights* L = &W->layers[l];
    size_t D = cfg->d_model;
    size_t KDh = (size_t)cfg->n_kv_heads * (size_t)cfg->head_dim; // 1024
    size_t F = cfg->ffn_hidden;

    L->rms1_weight = dev_alloc(D * cfg->bytes_per_elem);
    L->wq = dev_alloc(D * D * cfg->bytes_per_elem);
    L->wk = dev_alloc(D * KDh * cfg->bytes_per_elem);
    L->wv = dev_alloc(D * KDh * cfg->bytes_per_elem);
    L->wo = dev_alloc(D * D * cfg->bytes_per_elem);

    L->rms2_weight = dev_alloc(D * cfg->bytes_per_elem);
    L->w1 = dev_alloc(D * F * cfg->bytes_per_elem);
    L->w3 = dev_alloc(D * F * cfg->bytes_per_elem);
    L->w2 = dev_alloc(F * D * cfg->bytes_per_elem);
  }
  return 0;
}

void free_weights(ModelWeights* W) {
  if (!W) return;
  auto free_dev = [](void* p){ if (p) cudaFree(p); };

  free_dev(W->wte);
  free_dev(W->rms_final);
  free_dev(W->lm_head);

  if (W->layers) {
    // NOTE: if you later pack weights into one slab, change this.
    // For now, just free what we allocated.
    // (We don’t have cfg here, so leave as-is or store cfg in W if you want.)
  }
}
