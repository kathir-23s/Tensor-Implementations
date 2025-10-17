#include <cmath>
#include <stdexcept>
#include <iostream>
#include "UnaryOps/exp_log.hpp"
#include "Tensor.h"
#include "Types.h"

// for f16 and bf16
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {
// function pointers
static inline __device__ float expf_fn(float x){return expf(x);}
static inline __device__ double exp_fn(double x){return exp(x);}
static inline __device__ float exp2f_fn(float x){return exp2f(x);}
static inline __device__ double exp2_fn(double x){return exp2(x);}
static inline __device__ float logf_fn(float x){return logf(x);}
static inline __device__ double log_fn(double x){return log(x);}
static inline __device__ float log2f_fn(float x){return log2f(x);}
static inline __device__ double log2_fn(double x){return log2(x);}
static inline __device__ float log10f_fn(float x){return log10f(x);}
static inline __device__ double log10_fn(double x){return log10(x);}

// CUDA unary kernel
template<typename T_In, typename T_Out, T_Out(*Func)(T_Out)>
__global__
void unary_kernel_gpu(const T_In* in, T_Out* out, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        T_Out temp_val = static_cast<T_Out>(in[i]);
        out[i] = Func(temp_val);
    }
}
// Device kernel
template<typename T_Out, T_Out(*Func)(T_Out)>
__global__ void unary_half_kernel_gpu(const half* in, half* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float f32_val = __half2float(in[idx]);   // convert to f32
        f32_val = Func(f32_val);                 // compute exp
        out[idx] = __float2half(f32_val);        // convert back to half
    }
}

template<typename T_Out, T_Out(*Func)(T_Out)>
__global__ void unary_bfloat16_kernel_gpu(const __nv_bfloat16* in, __nv_bfloat16* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float f32_val = __bfloat162float(in[idx]);
        f32_val = Func(f32_val);
        out[idx] = __float2bfloat16(f32_val);
    }
}

// wrappers
Tensor exp_out_gpu_wrap(const Tensor& input_tensor){
    // assumes that the tensor already resides on GPU <====================================================================
    Dtype in_dtype = input_tensor.dtype();
    // fallback path for float32 or float64, etc.
    size_t size = input_tensor.numel();
    Tensor output_tensor(input_tensor.shape(), in_dtype,
                         input_tensor.device(), input_tensor.requires_grad());
    // launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // type promotion
    switch(in_dtype){
        case Dtype::Int16: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int16_t,float,expf_fn><<<blocks, threads>>>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int32_t,float,expf_fn><<<blocks, threads>>>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int64_t,double,exp_fn><<<blocks, threads>>>(
                input_tensor.data<int64_t>(),
                output_tensor.data<double>(),
                input_tensor.numel()
            );
            return output_tensor;
        }

    }
    // for float
    if (input_tensor.dtype() == Dtype::Float32) {
        const float* in_ptr = input_tensor.data<float>();
        float* out_ptr = output_tensor.data<float>();
        unary_kernel_gpu<float, float, expf_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else if (input_tensor.dtype() == Dtype::Float64) {
        const double* in_ptr = input_tensor.data<double>();
        double* out_ptr = output_tensor.data<double>();
        unary_kernel_gpu<double, double, exp_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } // New dispatch for half-precision types:
    else if (in_dtype == Dtype::Float16) {
    // convert input half → float32 for computation
    const half* in = input_tensor.data<half>();      // assuming float16_t == half
    half* out = output_tensor.data<half>();
    size_t size = input_tensor.numel();
    // Define launch configuration
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    // Launch kernel that internally converts and applies expf
    unary_half_kernel_gpu<float, expf_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else if (in_dtype == Dtype::Bfloat16) {
    const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
    __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
    size_t size = input_tensor.numel();
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    unary_bfloat16_kernel_gpu<float, expf_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else {
        throw std::runtime_error("Unsupported dtype for exp on CUDA");
    }
    cudaDeviceSynchronize();
    return output_tensor;
}
void exp_in_gpu_wrap(Tensor& input_tensor){
    try {
        Dtype dtype = input_tensor.dtype();
        size_t size = input_tensor.numel();
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        if (dtype == Dtype::Float32) {
            const float* in_cptr = input_tensor.data<float>();
            float* in_ptr = input_tensor.data<float>();
            unary_kernel_gpu<float, float, expf_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float64) {
            const double* in_cptr = input_tensor.data<double>();
            double* in_ptr = input_tensor.data<double>();
            unary_kernel_gpu<double, double, exp_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float16) {
            half* in_cptr = input_tensor.data<half>();
            half* in_ptr = input_tensor.data<half>();
            unary_half_kernel_gpu<float, expf_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Bfloat16) {
            __nv_bfloat16* in_cptr = input_tensor.data<__nv_bfloat16>();
            __nv_bfloat16* in_ptr = input_tensor.data<__nv_bfloat16>();
            unary_bfloat16_kernel_gpu<float, expf_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
            throw std::runtime_error("Error: cannot do inplace exp for Int data types!");
        } else {
            throw std::runtime_error("Unsupported dtype for inplace exp on CUDA!");
        }
        // Wait for GPU to finish
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught runtime error in exp_in_gpu_wrap: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

Tensor exp2_out_gpu_wrap(const Tensor& input_tensor){
    // assumes that the tensor already resides on GPU <====================================================================
    Dtype in_dtype = input_tensor.dtype();
    // fallback path for float32 or float64, etc.
    size_t size = input_tensor.numel();
    Tensor output_tensor(input_tensor.shape(), in_dtype,
                         input_tensor.device(), input_tensor.requires_grad());
    // launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // type promotion
    switch(in_dtype){
        case Dtype::Int16: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int16_t,float,exp2f_fn><<<blocks, threads>>>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int32_t,float,exp2f_fn><<<blocks, threads>>>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int64_t,double,exp2_fn><<<blocks, threads>>>(
                input_tensor.data<int64_t>(),
                output_tensor.data<double>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
    }
    // for float
    if (input_tensor.dtype() == Dtype::Float32) {
        const float* in_ptr = input_tensor.data<float>();
        float* out_ptr = output_tensor.data<float>();
        unary_kernel_gpu<float, float, exp2f_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else if (input_tensor.dtype() == Dtype::Float64) {
        const double* in_ptr = input_tensor.data<double>();
        double* out_ptr = output_tensor.data<double>();
        unary_kernel_gpu<double, double, exp2_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } // New dispatch for half-precision types:
    else if (in_dtype == Dtype::Float16) {
    // convert input half → float32 for computation
    const half* in = input_tensor.data<half>();      // assuming float16_t == half
    half* out = output_tensor.data<half>();
    size_t size = input_tensor.numel();
    // Define launch configuration
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    // Launch kernel that internally converts and applies expf
    unary_half_kernel_gpu<float, exp2f_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else if (in_dtype == Dtype::Bfloat16) {
    const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
    __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
    size_t size = input_tensor.numel();
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    unary_bfloat16_kernel_gpu<float, exp2f_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else {
        throw std::runtime_error("Unsupported dtype for exp on CUDA");
    }
    cudaDeviceSynchronize();
    return output_tensor;
}
void exp2_in_gpu_wrap(Tensor& input_tensor){
    try {
        Dtype dtype = input_tensor.dtype();
        size_t size = input_tensor.numel();
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        if (dtype == Dtype::Float32) {
            const float* in_cptr = input_tensor.data<float>();
            float* in_ptr = input_tensor.data<float>();
            unary_kernel_gpu<float, float, exp2f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float64) {
            const double* in_cptr = input_tensor.data<double>();
            double* in_ptr = input_tensor.data<double>();
            unary_kernel_gpu<double, double, exp2_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float16) {
            half* in_cptr = input_tensor.data<half>();
            half* in_ptr = input_tensor.data<half>();
            unary_half_kernel_gpu<float, exp2f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Bfloat16) {
            __nv_bfloat16* in_cptr = input_tensor.data<__nv_bfloat16>();
            __nv_bfloat16* in_ptr = input_tensor.data<__nv_bfloat16>();
            unary_bfloat16_kernel_gpu<float, exp2f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
            throw std::runtime_error("Error: cannot do inplace exp for Int data types!");
        } else {
            throw std::runtime_error("Unsupported dtype for inplace exp on CUDA!");
        }
        // Wait for GPU to finish
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught runtime error in exp_in_gpu_wrap: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

Tensor log_out_gpu_wrap(const Tensor& input_tensor){
    // assumes that the tensor already resides on GPU <====================================================================
    Dtype in_dtype = input_tensor.dtype();
    // fallback path for float32 or float64, etc.
    size_t size = input_tensor.numel();
    Tensor output_tensor(input_tensor.shape(), in_dtype,
                         input_tensor.device(), input_tensor.requires_grad());
    // launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // type promotion
    switch(in_dtype){
        case Dtype::Int16: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int16_t,float,logf_fn><<<blocks, threads>>>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int32_t,float,logf_fn><<<blocks, threads>>>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int64_t,double,log_fn><<<blocks, threads>>>(
                input_tensor.data<int64_t>(),
                output_tensor.data<double>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
    }
    // for float
    if (input_tensor.dtype() == Dtype::Float32) {
        const float* in_ptr = input_tensor.data<float>();
        float* out_ptr = output_tensor.data<float>();
        unary_kernel_gpu<float, float, logf_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else if (input_tensor.dtype() == Dtype::Float64) {
        const double* in_ptr = input_tensor.data<double>();
        double* out_ptr = output_tensor.data<double>();
        unary_kernel_gpu<double, double, log_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } // New dispatch for half-precision types:
    else if (in_dtype == Dtype::Float16) {
    // convert input half → float32 for computation
    const half* in = input_tensor.data<half>();      // assuming float16_t == half
    half* out = output_tensor.data<half>();
    size_t size = input_tensor.numel();
    // Define launch configuration
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    // Launch kernel that internally converts and applies expf
    unary_half_kernel_gpu<float, logf_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else if (in_dtype == Dtype::Bfloat16) {
    const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
    __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
    size_t size = input_tensor.numel();
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    unary_bfloat16_kernel_gpu<float, logf_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else {
        throw std::runtime_error("Unsupported dtype for exp on CUDA");
    }
    cudaDeviceSynchronize();
    return output_tensor;
}
void log_in_gpu_wrap(Tensor& input_tensor){
    try {
        Dtype dtype = input_tensor.dtype();
        size_t size = input_tensor.numel();
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        if (dtype == Dtype::Float32) {
            const float* in_cptr = input_tensor.data<float>();
            float* in_ptr = input_tensor.data<float>();
            unary_kernel_gpu<float, float, logf_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float64) {
            const double* in_cptr = input_tensor.data<double>();
            double* in_ptr = input_tensor.data<double>();
            unary_kernel_gpu<double, double, log_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float16) {
            half* in_cptr = input_tensor.data<half>();
            half* in_ptr = input_tensor.data<half>();
            unary_half_kernel_gpu<float, logf_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Bfloat16) {
            __nv_bfloat16* in_cptr = input_tensor.data<__nv_bfloat16>();
            __nv_bfloat16* in_ptr = input_tensor.data<__nv_bfloat16>();
            unary_bfloat16_kernel_gpu<float, logf_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
            throw std::runtime_error("Error: cannot do inplace exp for Int data types!");
        } else {
            throw std::runtime_error("Unsupported dtype for inplace exp on CUDA!");
        }
        // Wait for GPU to finish
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught runtime error in exp_in_gpu_wrap: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

Tensor log2_out_gpu_wrap(const Tensor& input_tensor){
    // assumes that the tensor already resides on GPU <====================================================================
    Dtype in_dtype = input_tensor.dtype();
    // fallback path for float32 or float64, etc.
    size_t size = input_tensor.numel();
    Tensor output_tensor(input_tensor.shape(), in_dtype,
                         input_tensor.device(), input_tensor.requires_grad());
    // launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // type promotion
    switch(in_dtype){
        case Dtype::Int16: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int16_t,float,log2f_fn><<<blocks, threads>>>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int32_t,float,log2f_fn><<<blocks, threads>>>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int64_t,double,log2_fn><<<blocks, threads>>>(
                input_tensor.data<int64_t>(),
                output_tensor.data<double>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
    }
    // for float
    if (input_tensor.dtype() == Dtype::Float32) {
        const float* in_ptr = input_tensor.data<float>();
        float* out_ptr = output_tensor.data<float>();
        unary_kernel_gpu<float, float, log2f_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else if (input_tensor.dtype() == Dtype::Float64) {
        const double* in_ptr = input_tensor.data<double>();
        double* out_ptr = output_tensor.data<double>();
        unary_kernel_gpu<double, double, log2_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } // New dispatch for half-precision types:
    else if (in_dtype == Dtype::Float16) {
    // convert input half → float32 for computation
    const half* in = input_tensor.data<half>();      // assuming float16_t == half
    half* out = output_tensor.data<half>();
    size_t size = input_tensor.numel();
    // Define launch configuration
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    // Launch kernel that internally converts and applies expf
    unary_half_kernel_gpu<float, log2f_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else if (in_dtype == Dtype::Bfloat16) {
    const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
    __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
    size_t size = input_tensor.numel();
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    unary_bfloat16_kernel_gpu<float, log2f_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else {
        throw std::runtime_error("Unsupported dtype for exp on CUDA");
    }
    cudaDeviceSynchronize();
    return output_tensor;
}
void log2_in_gpu_wrap(Tensor& input_tensor){
    try {
        Dtype dtype = input_tensor.dtype();
        size_t size = input_tensor.numel();
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        if (dtype == Dtype::Float32) {
            const float* in_cptr = input_tensor.data<float>();
            float* in_ptr = input_tensor.data<float>();
            unary_kernel_gpu<float, float, log2f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float64) {
            const double* in_cptr = input_tensor.data<double>();
            double* in_ptr = input_tensor.data<double>();
            unary_kernel_gpu<double, double, log2_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float16) {
            half* in_cptr = input_tensor.data<half>();
            half* in_ptr = input_tensor.data<half>();
            unary_half_kernel_gpu<float, log2f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Bfloat16) {
            __nv_bfloat16* in_cptr = input_tensor.data<__nv_bfloat16>();
            __nv_bfloat16* in_ptr = input_tensor.data<__nv_bfloat16>();
            unary_bfloat16_kernel_gpu<float, log2f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
            throw std::runtime_error("Error: cannot do inplace exp for Int data types!");
        } else {
            throw std::runtime_error("Unsupported dtype for inplace exp on CUDA!");
        }
        // Wait for GPU to finish
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught runtime error in exp_in_gpu_wrap: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

Tensor log10_out_gpu_wrap(const Tensor& input_tensor){
    // assumes that the tensor already resides on GPU <====================================================================
    Dtype in_dtype = input_tensor.dtype();
    // fallback path for float32 or float64, etc.
    size_t size = input_tensor.numel();
    Tensor output_tensor(input_tensor.shape(), in_dtype,
                         input_tensor.device(), input_tensor.requires_grad());
    // launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // type promotion
    switch(in_dtype){
        case Dtype::Int16: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int16_t,float,log10f_fn><<<blocks, threads>>>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int32_t,float,log10f_fn><<<blocks, threads>>>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_gpu<int64_t,double,log10_fn><<<blocks, threads>>>(
                input_tensor.data<int64_t>(),
                output_tensor.data<double>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
    }
    // for float
    if (input_tensor.dtype() == Dtype::Float32) {
        const float* in_ptr = input_tensor.data<float>();
        float* out_ptr = output_tensor.data<float>();
        unary_kernel_gpu<float, float, log10f_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else if (input_tensor.dtype() == Dtype::Float64) {
        const double* in_ptr = input_tensor.data<double>();
        double* out_ptr = output_tensor.data<double>();
        unary_kernel_gpu<double, double, log10_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } // New dispatch for half-precision types:
    else if (in_dtype == Dtype::Float16) {
    // convert input half → float32 for computation
    const half* in = input_tensor.data<half>();      // assuming float16_t == half
    half* out = output_tensor.data<half>();
    size_t size = input_tensor.numel();
    // Define launch configuration
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    // Launch kernel that internally converts and applies expf
    unary_half_kernel_gpu<float, log10f_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else if (in_dtype == Dtype::Bfloat16) {
    const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
    __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
    size_t size = input_tensor.numel();
    dim3 threads(256);
    dim3 blocks((size + threads.x - 1) / threads.x);
    unary_bfloat16_kernel_gpu<float, log10f_fn><<<blocks, threads>>>(in, out, size);
    cudaDeviceSynchronize();
    } else {
        throw std::runtime_error("Unsupported dtype for exp on CUDA");
    }
    cudaDeviceSynchronize();
    return output_tensor;
}
void log10_in_gpu_wrap(Tensor& input_tensor){
    try {
        Dtype dtype = input_tensor.dtype();
        size_t size = input_tensor.numel();
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        if (dtype == Dtype::Float32) {
            const float* in_cptr = input_tensor.data<float>();
            float* in_ptr = input_tensor.data<float>();
            unary_kernel_gpu<float, float, log10f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float64) {
            const double* in_cptr = input_tensor.data<double>();
            double* in_ptr = input_tensor.data<double>();
            unary_kernel_gpu<double, double, log10_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Float16) {
            half* in_cptr = input_tensor.data<half>();
            half* in_ptr = input_tensor.data<half>();
            unary_half_kernel_gpu<float, log10f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Bfloat16) {
            __nv_bfloat16* in_cptr = input_tensor.data<__nv_bfloat16>();
            __nv_bfloat16* in_ptr = input_tensor.data<__nv_bfloat16>();
            unary_bfloat16_kernel_gpu<float, log10f_fn><<<blocks, threads>>>(in_cptr, in_ptr, size);
        } else if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
            throw std::runtime_error("Error: cannot do inplace exp for Int data types!");
        } else {
            throw std::runtime_error("Unsupported dtype for inplace exp on CUDA!");
        }
        // Wait for GPU to finish
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught runtime error in exp_in_gpu_wrap: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
}