#include "model_config.h"
#include "weights.h"
#include "kv_cache.h"
#include "graph.h"
#include "ops.h"
#include "tensor.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdio.h>
#include <stdlib.h>

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s : %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}

static void init_handles(RuntimeHandles* h) {
  ck(cudaStreamCreate(&h->stream), "stream");
  cublasCreate(&h->cublas);
  cublasLtCreate(&h->cublasLt);
  cublasSetStream(h->cublas, h->stream);
}

static void destroy_handles(RuntimeHandles* h) {
  cublasDestroy(h->cublas);
  cublasLtDestroy(h->cublasLt);
  cudaStreamDestroy(h->stream);
}

int main() {
  // --- Config (fill from your config.json) ---
  ModelConfig cfg = {};
  cfg.n_layers = 32;
  cfg.d_model = 4096;
  cfg.n_heads = 32;
  cfg.head_dim = 128;
  cfg.n_kv_heads = 8;
  cfg.ffn_hidden = 14336;
  cfg.vocab_size = 32000;     // TODO
  cfg.max_context = 8192;     // TODO verify
  cfg.sliding_window = 4096;  // TODO verify
  cfg.rope_theta = 1e6f;      // TODO verify
  cfg.bytes_per_elem = 2;     // fp16/bf16

  RuntimeHandles h = {};
  init_handles(&h);

  // --- Graph (debug/visual) ---
  ModelGraph g = {};
  build_graph_empty(&g);
  print_graph(&g);

  // --- Weights (empty allocate) ---
  ModelWeights W = {};
  load_weights_empty("./weights", &cfg, &W, &h);

  // --- KV Cache ---
  const int B = 1;
  KVPagedCache kv = {};
  kv_cache_init_empty(&kv, &cfg, B, cfg.max_context);

  // --- Inputs ---
  const int S_prompt = 128; // TODO
  int32_t* tokens_dev = NULL;
  ck(cudaMalloc(&tokens_dev, sizeof(int32_t) * B * S_prompt), "tokens_dev");

  // --- Hidden states ---
  Tensor X = {};
  // TODO: allocate X.data device: [B,S,D]
  // ck(cudaMalloc(&X.data, ...), "X");

  // Embed
  op_embed_empty(&X, tokens_dev, B, S_prompt, &W, &cfg, &h);

  // Prefill through all layers
  for (int l = 0; l < cfg.n_layers; l++) {
    Tensor Y = {};
    op_layer_prefill_empty(&Y, &X, l, B, S_prompt, /*pos0=*/0, &W, &kv, &cfg, &h);
    X = Y; // swap (you’ll manage buffers properly)
  }
  kv.cur_T = S_prompt; // placeholder

  // LM head -> logits for last token
  Tensor logits = {};
  op_lm_head_empty(&logits, &X, &W, &cfg, &h);

  // Copy logits to host and sample
  float* logits_host = (float*)malloc(sizeof(float) * cfg.vocab_size);
  // TODO: cudaMemcpyAsync logits for last position to host
  // ck(cudaMemcpyAsync(...), "logits copy");
  ck(cudaStreamSynchronize(h.stream), "sync");

  int32_t next = 0; // sample_next_token_empty(logits_host, cfg.vocab_size, 1.0f);

  // Decode loop
  const int max_new_tokens = 16;
  for (int t = 0; t < max_new_tokens; t++) {
    int pos = kv.cur_T;

    // x is embedding of the next token (or carry X and run embed for S=1)
    // TODO: build x:[B,1,D]
    Tensor x = {};
    // TODO

    for (int l = 0; l < cfg.n_layers; l++) {
      Tensor y = {};
      op_layer_decode_step_empty(&y, &x, l, B, pos, &W, &kv, &cfg, &h);
      x = y;
    }

    kv.cur_T++;

    Tensor logits1 = {};
    op_lm_head_empty(&logits1, &x, &W, &cfg, &h);

    // TODO: copy logits1 -> host, sample -> next
    next = 0;
  }

  free(logits_host);
  cudaFree(tokens_dev);

  free_graph(&g);
  destroy_handles(&h);
  return 0;
}
