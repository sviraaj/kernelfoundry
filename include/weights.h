#pragma once
#include "model_config.h"
#include <cublas_v2.h>
#include <cublasLt.h>

typedef struct LayerWeights {
  // Attention
  void* wq; // [D, D]
  void* wk; // [D, K*Dh] = [4096, 1024]
  void* wv; // [D, 1024]
  void* wo; // [D, D]
  void* rms1_weight; // [D]

  // MLP (SwiGLU)
  void* w1; // [D, F]
  void* w3; // [D, F]
  void* w2; // [F, D]
  void* rms2_weight; // [D]
} LayerWeights;

typedef struct ModelWeights {
  void* wte;        // [V, D]
  void* rms_final;  // [D]
  void* lm_head;    // [D, V] (or null if tied with wte)
  LayerWeights* layers; // n_layers
} ModelWeights;

typedef struct RuntimeHandles {
  cublasHandle_t cublas;
  cublasLtHandle_t cublasLt;
  cudaStream_t stream;
} RuntimeHandles;

// You populate these:
int load_weights_empty(const char* path, const ModelConfig* cfg, ModelWeights* W, RuntimeHandles* h);
void free_weights(ModelWeights* W);
