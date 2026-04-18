#include "graph.h"
#include <stdio.h>
#include <stdlib.h>

int build_graph_empty(ModelGraph* g) {
  // Minimal illustrative “graph” (not per-layer expanded).
  // You can expand this into per-layer subgraphs or keep it symbolic.
  g->n_nodes = 8;
  g->nodes = (GraphNode*)calloc(g->n_nodes, sizeof(GraphNode));
  if (!g->nodes) return -1;

  int i = 0;
  g->nodes[i++] = (GraphNode){ "Embed", NODE_EMBED, {-1}, 0, {0}, 1 };
  g->nodes[i++] = (GraphNode){ "Block[0..31]: RMS1->QKV->RoPE->KV->Attn->O->Res", NODE_ATTENTION, {0}, 1, {1}, 1 };
  g->nodes[i++] = (GraphNode){ "Block[0..31]: RMS2->W1/W3->SwiGLU->W2->Res", NODE_MLP_SWIGLU, {1}, 1, {2}, 1 };
  g->nodes[i++] = (GraphNode){ "Final RMSNorm", NODE_FINAL_RMSNORM, {2}, 1, {3}, 1 };
  g->nodes[i++] = (GraphNode){ "LM Head", NODE_LM_HEAD, {3}, 1, {4}, 1 };
  g->nodes[i++] = (GraphNode){ "Sampling", NODE_SAMPLING, {4}, 1, {5}, 1 };

  return 0;
}

void free_graph(ModelGraph* g) {
  if (g && g->nodes) free(g->nodes);
  g->nodes = NULL;
  g->n_nodes = 0;
}

void print_graph(const ModelGraph* g) {
  printf("ModelGraph: %d nodes\n", g->n_nodes);
  for (int i = 0; i < g->n_nodes; i++) {
    printf("  [%d] %s (type=%d) inputs=%d outputs=%d\n",
           i, g->nodes[i].name, (int)g->nodes[i].type, g->nodes[i].n_inputs, g->nodes[i].n_outputs);
  }
}
