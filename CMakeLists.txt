cmake_minimum_required(VERSION 3.20)
project(kernelfoundry LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Override at configure time: cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()
add_subdirectory(tests)
