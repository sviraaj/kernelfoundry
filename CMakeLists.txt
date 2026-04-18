cmake_minimum_required(VERSION 3.20)
project(warpforge LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default to a broad set of modern architectures.
# Override at configure time: cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(warpforge
  src/main.cu
  src/graph.cu
  src/weights.cu
  src/kv_cache.cu
  src/ops.cu
  src/kernels.cu
)

target_include_directories(warpforge PRIVATE include)

target_link_libraries(warpforge PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
)

# Uncomment for device-side line info (useful with cuda-gdb / Nsight):
# target_compile_options(warpforge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# ---------- Tests ----------
option(BUILD_TESTS "Build kernel unit tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
