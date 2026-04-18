#pragma once
#include <stdint.h>

typedef enum NodeType {
  NODE_EMBED = 0,
  NODE_RMSNORM,
  NODE_QKV_PROJ,
  NODE_ROPE,
  NODE_KV_APPEND,
  NODE_ATTENTION,
  NODE_O_PROJ,
  NODE_RESIDUAL,
  NODE_MLP_SWIGLU,
  NODE_FINAL_RMSNORM,
  NODE_LM_HEAD,
  NODE_SAMPLING
} NodeType;

typedef struct GraphNode {
  const char* name;
  NodeType type;
  int32_t inputs[4];
  int32_t n_inputs;
  int32_t outputs[2];
  int32_t n_outputs;
} GraphNode;

typedef struct ModelGraph {
  GraphNode* nodes;
  int32_t n_nodes;
} ModelGraph;

int build_graph_empty(ModelGraph* g);
void free_graph(ModelGraph* g);
void print_graph(const ModelGraph* g);
