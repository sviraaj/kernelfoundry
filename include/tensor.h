#pragma once
#include <stddef.h>
#include <stdint.h>

typedef enum DType {
  DTYPE_F16 = 0,
  DTYPE_BF16 = 1,
  DTYPE_F32 = 2,
  DTYPE_I32 = 3
} DType;

typedef struct Tensor {
  void*  data;      // device pointer unless noted
  int64_t shape[4]; // up to 4D for this scaffold
  int32_t ndims;
  DType dtype;
  size_t bytes;
} Tensor;

static inline int64_t numel(const Tensor* t) {
  int64_t n = 1;
  for (int i = 0; i < t->ndims; i++) n *= t->shape[i];
  return n;
}
