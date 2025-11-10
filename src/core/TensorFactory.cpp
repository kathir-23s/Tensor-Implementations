#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include <random>
#include "device/DeviceCore.h"//✨✨✨

using namespace OwnTensor;

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <curand.h>

// Helper for CUDA RNG
void cuda_rand_uniform(float* data, size_t count, unsigned long seed, cudaStream_t stream) {//✨✨✨
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetStream(gen, stream);//✨✨✨
    curandGenerateUniform(gen, data, count);
    curandDestroyGenerator(gen);
}

void cuda_rand_uniform(double* data, size_t count, unsigned long seed, cudaStream_t stream) {//✨✨✨
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetStream(gen, stream);//✨✨✨
    curandGenerateUniformDouble(gen, data, count);
    curandDestroyGenerator(gen);
}

void cuda_rand_normal(float* data, size_t count, unsigned long seed, cudaStream_t stream) {//✨✨✨
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetStream(gen, stream);//✨✨✨
    curandGenerateNormal(gen, data, count, 0.0f, 1.0f);
    curandDestroyGenerator(gen);
}

void cuda_rand_normal(double* data, size_t count, unsigned long seed, cudaStream_t stream) {//✨✨✨
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetStream(gen, stream);//✨✨✨
    curandGenerateNormalDouble(gen, data, count, 0.0, 1.0);
    curandDestroyGenerator(gen);
}
#endif

Tensor Tensor::zeros(Shape shape, TensorOptions opts) {
    Tensor tensor(shape, opts);
    
    if (opts.device.is_cpu()) {
        // CPU implementation - handles all 7 types automatically
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            tensor.fill(T(0));
        });
    } else {
        // GPU implementation - optimized with cudaMemset
#ifdef WITH_CUDA
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
        cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);//✨✨✨
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    return tensor;
}

Tensor Tensor::ones(Shape shape, TensorOptions opts) {
    Tensor tensor(shape, opts);
    
    if (opts.device.is_cpu()) {
        // CPU implementation - handles all 7 types automatically
        // dispatch_by_dtype(opts.dtype, [&](auto dummy) {
        //     using T = decltype(dummy);
        //     tensor.fill(T(1));
        // });
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, bool>) {
                // Special handling for bool: ones = true
                tensor.fill(true);
            } else {
                tensor.fill(T(1));
            }
        });
    } else {
        // GPU implementation - handles all 7 types automatically
#ifdef WITH_CUDA
        [[maybe_unused]] cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, bool>) {
                // For bool on GPU, use memset with 1
                cudaMemset(tensor.data(), 1, tensor.numel());
            } else {
                std::vector<T> ones_data(tensor.numel(), T(1));
                cudaMemcpy(tensor.data(), ones_data.data(),
                          tensor.numel() * sizeof(T), cudaMemcpyHostToDevice);
            }
        });
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    return tensor;
}

Tensor Tensor::full(Shape shape, TensorOptions opts, float value) {
    Tensor tensor(shape, opts);
    
    if (opts.device.is_cpu()) {
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, bool>) {
                // For bool: any nonzero value = true
                tensor.fill(value != 0.0f);
            } else {
                tensor.fill(static_cast<T>(value));
            }
        });
    } else {
#ifdef WITH_CUDA
        [[maybe_unused]] cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, bool>) {
                uint8_t bool_val = (value != 0.0f) ? 1 : 0;
                cudaMemset(tensor.data(), bool_val, tensor.numel());
            } else {
                std::vector<T> fill_data(tensor.numel(), static_cast<T>(value));
                cudaMemcpy(tensor.data(), fill_data.data(),
                          tensor.numel() * sizeof(T), cudaMemcpyHostToDevice);
            }
        });
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
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
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
        
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, float>) {
                cuda_rand_uniform(static_cast<float*>(tensor.data()), tensor.numel(), seed, stream);//✨✨✨
            } else if constexpr (std::is_same_v<T, double>) {
                cuda_rand_uniform(static_cast<double*>(tensor.data()), tensor.numel(), seed, stream);//✨✨✨
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
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
        
        dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_same_v<T, float>) {
                cuda_rand_normal(static_cast<float*>(tensor.data()), tensor.numel(), seed, stream);//✨✨✨
            } else if constexpr (std::is_same_v<T, double>) {
                cuda_rand_normal(static_cast<double*>(tensor.data()), tensor.numel(), seed, stream);//✨✨✨
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

