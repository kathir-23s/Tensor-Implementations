#include "Tensor.h"
#include "TensorDispatch.h"
#include <random>

using namespace OwnTensor;

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <curand.h>

// Helper for CUDA RNG
void cuda_rand_uniform(float* data, size_t count, unsigned long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, data, count);
    curandDestroyGenerator(gen);
}

void cuda_rand_uniform(double* data, size_t count, unsigned long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, data, count);
    curandDestroyGenerator(gen);
}

void cuda_rand_normal(float* data, size_t count, unsigned long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, data, count, 0.0f, 1.0f);
    curandDestroyGenerator(gen);
}

void cuda_rand_normal(double* data, size_t count, unsigned long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormalDouble(gen, data, count, 0.0, 1.0);
    curandDestroyGenerator(gen);
}
#endif

Tensor Tensor::zeros(Shape shape, TensorOptions opts) {
    Tensor tensor(shape, opts);
    dispatch_by_dtype(opts.dtype, [&](auto dummy) {
        using T = decltype(dummy);
        tensor.fill(T(0));
    });
    return tensor;
}

Tensor Tensor::ones(Shape shape, TensorOptions opts) {
    Tensor tensor(shape, opts);
    dispatch_by_dtype(opts.dtype, [&](auto dummy) {
        using T = decltype(dummy);
        tensor.fill(T(1));
    });
    return tensor;
}

Tensor Tensor::full(Shape shape, TensorOptions opts, float value) {
    Tensor tensor(shape, opts);
    dispatch_by_dtype(opts.dtype, [&](auto dummy) {
        using T = decltype(dummy);
        tensor.fill(static_cast<T>(value));
    });
    return tensor;
}

Tensor Tensor::rand(Shape shape, TensorOptions opts) {
    Tensor tensor(shape, opts);
    
    if (opts.device.is_cpu()) {
        // CPU random
        std::random_device rd;
        std::mt19937 gen(rd());
        
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dist(0.0, 1.0);
                T* data = static_cast<T*>(tensor.data());
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    data[i] = dist(gen);
                }
            } else {
                throw std::runtime_error("rand only supports floating point types");
            }
        });
    } else {
        // GPU random
#ifdef WITH_CUDA
        std::random_device rd;
        unsigned long seed = rd();
        
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, float>) {
                cuda_rand_uniform(static_cast<float*>(tensor.data()), tensor.numel(), seed);
            } else if constexpr (std::is_same_v<T, double>) {
                cuda_rand_uniform(static_cast<double*>(tensor.data()), tensor.numel(), seed);
            } else {
                throw std::runtime_error("GPU rand only supports float/double");
            }
        });
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return tensor;
}

Tensor Tensor::randn(Shape shape, TensorOptions opts) {
    Tensor tensor(shape, opts);
    
    if (opts.device.is_cpu()) {
        // CPU random
        std::random_device rd;
        std::mt19937 gen(rd());
        
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_floating_point_v<T>) {
                std::normal_distribution<T> dist(0.0, 1.0);
                T* data = static_cast<T*>(tensor.data());
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    data[i] = dist(gen);
                }
            } else {
                throw std::runtime_error("randn only supports floating point types");
            }
        });
    } else {
        // GPU random
#ifdef WITH_CUDA
        std::random_device rd;
        unsigned long seed = rd();
        
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, float>) {
                cuda_rand_normal(static_cast<float*>(tensor.data()), tensor.numel(), seed);
            } else if constexpr (std::is_same_v<T, double>) {
                cuda_rand_normal(static_cast<double*>(tensor.data()), tensor.numel(), seed);
            } else {
                throw std::runtime_error("GPU randn only supports float/double");
            }
        });
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return tensor;
}

