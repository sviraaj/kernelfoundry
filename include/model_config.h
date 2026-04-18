#pragma once
#include <stdint.h>

typedef struct ModelConfig {
  int32_t n_layers;      // 32
  int32_t d_model;       // 4096
  int32_t n_heads;       // 32
  int32_t head_dim;      // 128
  int32_t n_kv_heads;    // 8 (GQA)
  int32_t ffn_hidden;    // 14336 (SwiGLU)
  int32_t vocab_size;    // fill from config.json
  int32_t max_context;   // often 8192
  int32_t sliding_window;// often 4096 (0 or -1 means full)
  float   rope_theta;    // e.g., 1e6

  // dtype sizes (you can switch)
  int32_t bytes_per_elem; // 2 for fp16/bf16
} ModelConfig;

