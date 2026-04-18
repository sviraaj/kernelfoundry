#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(err)                                                          \
    do {                                                                         \
        cudaError_t _e = (err);                                                  \
        if (_e != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(_e) << '\n';               \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

inline float max_abs_error(const std::vector<float>& ref, const std::vector<float>& got) {
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_err = std::max(max_err, std::fabs(ref[i] - got[i]));
    return max_err;
}

inline void fill_random_f32(std::vector<float>& v, float lo, float hi, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : v) x = dist(rng);
}

inline float* alloc_device_f32(int n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

inline void upload_f32(float* d, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
}

inline void download_f32(std::vector<float>& h, const float* d, int n) {
    h.resize(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
}
