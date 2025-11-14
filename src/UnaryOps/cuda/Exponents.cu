#include <iostream>
#include <stdexcept>
#include <cmath>
#include "ops/helpers/exp_log.hpp"
#include "core/Tensor.h"
#include "dtype/Types.h"

// For f16 and bf16
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {

// ============================================================================
// Device Function Pointers for GPU Math Operations
// ============================================================================
static inline __device__ float expf_fn(float x) { return expf(x); }
static inline __device__ double exp_fn(double x) { return exp(x); }
static inline __device__ float exp2f_fn(float x) { return exp2f(x); }
static inline __device__ double exp2_fn(double x) { return exp2(x); }
static inline __device__ float logf_fn(float x) { return logf(x); }
static inline __device__ double log_fn(double x) { return log(x); }
static inline __device__ float log2f_fn(float x) { return log2f(x); }
static inline __device__ double log2_fn(double x) { return log2(x); }
static inline __device__ float log10f_fn(float x) { return log10f(x); }
static inline __device__ double log10_fn(double x) { return log10(x); }

// ============================================================================
// Generic CUDA Unary Kernel (for standard types)
// ============================================================================
template<typename T_In, typename T_Out, T_Out(*Func)(T_Out)>
__global__ void unary_kernel_gpu(const T_In* in, T_Out* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T_Out temp_val = static_cast<T_Out>(in[idx]);
        out[idx] = Func(temp_val);
    }
}

// ============================================================================
// Specialized CUDA Kernel for Float16 (half precision)
// ============================================================================
template<float(*Func)(float)>
__global__ void unary_half_kernel_gpu(const half* in, half* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float f32_val = __half2float(in[idx]);
        f32_val = Func(f32_val);
        out[idx] = __float2half(f32_val);
    }
}

// ============================================================================
// Specialized CUDA Kernel for Bfloat16
// ============================================================================
template<float(*Func)(float)>
__global__ void unary_bfloat16_kernel_gpu(const __nv_bfloat16* in, __nv_bfloat16* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float f32_val = __bfloat162float(in[idx]);
        f32_val = Func(f32_val);
        out[idx] = __float2bfloat16(f32_val);
    }
}

// ============================================================================
// Helper: Determine output dtype for integer promotion
// ============================================================================
inline Dtype get_promoted_dtype(Dtype input_dtype) {
    switch(input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
            return Dtype::Float32;
        case Dtype::Int64:
            return Dtype::Float64;
        default:
            return input_dtype;
    }
}

// ============================================================================
// GPU-specific dispatch that ONLY handles int/float/double (no bf16/f16)
// ============================================================================
template<typename Func>
static auto dispatch_gpu_dtype(Dtype dtype, Func&& f) {
    switch(dtype) {
        // Integer types
        case Dtype::Int16: return f(int16_t{});
        case Dtype::Int32: return f(int32_t{});
        case Dtype::Int64: return f(int64_t{});
        //Boolean
        case Dtype::Bool: return f(uint8_t{});
        // Floating point types
        case Dtype::Float32: return f(float{});
        case Dtype::Float64: return f(double{});
        // Explicitly exclude bf16/f16 - they should be handled separately
        case Dtype::Float16:
        case Dtype::Bfloat16:
            throw std::runtime_error("Float16/Bfloat16 should be handled before dispatch!");
        default:
            throw std::runtime_error("Unsupported dtype in dispatch_gpu_dtype");
    }
}

// ============================================================================
// Generic Out-of-Place GPU Wrapper
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
Tensor generic_unary_out_gpu(const Tensor& input_tensor, cudaStream_t stream) {//✨✨✨
    Dtype in_dtype = input_tensor.dtype();
    size_t size = input_tensor.numel();
    
    // Launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Handle Float16 - uses CUDA native half type
    if (in_dtype == Dtype::Float16) {
        Tensor output_tensor(input_tensor.shape(), in_dtype, 
                           input_tensor.device(), input_tensor.requires_grad());
        const half* in = input_tensor.data<half>();
        half* out = output_tensor.data<half>();
        
        unary_half_kernel_gpu<FloatFunc><<<blocks, threads, 0, stream>>>(in, out, size);//✨✨✨
        //cudaDeviceSynchronize();//✨✨✨
        return output_tensor;
    }
    
    // Handle Bfloat16 - uses CUDA native bfloat16 type
    if (in_dtype == Dtype::Bfloat16) {
        Tensor output_tensor(input_tensor.shape(), in_dtype, 
                           input_tensor.device(), input_tensor.requires_grad());
        const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
        __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
        
        unary_bfloat16_kernel_gpu<FloatFunc><<<blocks, threads, 0, stream>>>(in, out, size);//✨✨✨
        //cudaDeviceSynchronize();//✨✨✨
        return output_tensor;
    }
    
    // Now dispatch only handles int/float/double types
    Dtype output_dtype = get_promoted_dtype(in_dtype);
    Tensor output_tensor(input_tensor.shape(), output_dtype, 
                        input_tensor.device(), input_tensor.requires_grad());
    
    // Use GPU-specific dispatch for input type
    dispatch_gpu_dtype(in_dtype, [&](auto in_type_instance) {
        using InputType = decltype(in_type_instance);
        
        // Use GPU-specific dispatch for output type
        dispatch_gpu_dtype(output_dtype, [&](auto out_type_instance) {
            using OutputType = decltype(out_type_instance);
            
            const InputType* in_ptr = input_tensor.data<InputType>();
            OutputType* out_ptr = output_tensor.data<OutputType>();
            
            // Select appropriate function based on output type
            if constexpr (std::is_same_v<OutputType, float>) {
                unary_kernel_gpu<InputType, OutputType, FloatFunc><<<blocks, threads, 0, stream>>>(//✨✨✨
                    in_ptr, out_ptr, size
                );
            } else if constexpr (std::is_same_v<OutputType, double>) {
                unary_kernel_gpu<InputType, OutputType, DoubleFunc><<<blocks, threads, 0, stream>>>(//✨✨✨
                    in_ptr, out_ptr, size
                );
            }
        });
    });
    
    //cudaDeviceSynchronize();//✨✨✨
    return output_tensor;
}

// ============================================================================
// Generic In-Place GPU Wrapper
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
void generic_unary_in_gpu(Tensor& input_tensor, cudaStream_t stream) {//✨✨✨
    Dtype dtype = input_tensor.dtype();
    size_t size = input_tensor.numel();
    
    // Reject integer types for in-place operations
    if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
        throw std::runtime_error("Error: cannot do inplace operations for Int data types!");
    }
    
    // Launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Handle Float16
    if (dtype == Dtype::Float16) {
        half* data_ptr = input_tensor.data<half>();
        unary_half_kernel_gpu<FloatFunc><<<blocks, threads, 0, stream>>>(data_ptr, data_ptr, size);//✨✨✨
        //cudaDeviceSynchronize();//✨✨✨
        return;
    }
    
    // Handle Bfloat16
    if (dtype == Dtype::Bfloat16) {
        __nv_bfloat16* data_ptr = input_tensor.data<__nv_bfloat16>();
        unary_bfloat16_kernel_gpu<FloatFunc><<<blocks, threads, 0, stream>>>(data_ptr, data_ptr, size);//✨✨✨
        //cudaDeviceSynchronize();//✨✨✨
        return;
    }
    
    // Now dispatch only handles float/double types
    dispatch_gpu_dtype(dtype, [&](auto type_instance) {
        using DataType = decltype(type_instance);
        
        DataType* data_ptr = input_tensor.data<DataType>();
        
        if constexpr (std::is_same_v<DataType, float>) {
            unary_kernel_gpu<DataType, DataType, FloatFunc><<<blocks, threads, 0, stream>>>(//✨✨✨
                data_ptr, data_ptr, size
            );
        } else if constexpr (std::is_same_v<DataType, double>) {
            unary_kernel_gpu<DataType, DataType, DoubleFunc><<<blocks, threads, 0, stream>>>(//✨✨✨
                data_ptr, data_ptr, size
            );
        }
    });
    
    //✨✨✨
    // Wait for GPU to finish
    // cudaError_t err = //cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
    // }//✨✨✨
}

// ============================================================================
// GPU Wrapper Functions - REMOVE TRY-CATCH BLOCKS
// ============================================================================
Tensor exp_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {//✨✨✨
    return generic_unary_out_gpu<expf_fn, exp_fn>(input, stream);//✨✨✨
}

void exp_in_gpu_wrap(Tensor& input, cudaStream_t stream) {//✨✨✨
    generic_unary_in_gpu<expf_fn, exp_fn>(input, stream);//✨✨✨
}

Tensor exp2_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {//✨✨✨
    return generic_unary_out_gpu<exp2f_fn, exp2_fn>(input, stream);//✨✨✨
}

void exp2_in_gpu_wrap(Tensor& input, cudaStream_t stream) {//✨✨✨
    generic_unary_in_gpu<exp2f_fn, exp2_fn>(input, stream);//✨✨✨
}

Tensor log_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {//✨✨✨
    return generic_unary_out_gpu<logf_fn, log_fn>(input, stream);//✨✨✨
}

void log_in_gpu_wrap(Tensor& input, cudaStream_t stream) {//✨✨✨
    generic_unary_in_gpu<logf_fn, log_fn>(input, stream);//✨✨✨
}

Tensor log2_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {//✨✨✨
    return generic_unary_out_gpu<log2f_fn, log2_fn>(input, stream);//✨✨✨
}

void log2_in_gpu_wrap(Tensor& input, cudaStream_t stream) {//✨✨✨
    generic_unary_in_gpu<log2f_fn, log2_fn>(input, stream);//✨✨✨
}

Tensor log10_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {//✨✨✨
    return generic_unary_out_gpu<log10f_fn, log10_fn>(input, stream);//✨✨✨
}

void log10_in_gpu_wrap(Tensor& input, cudaStream_t stream) {//✨✨✨
    generic_unary_in_gpu<log10f_fn, log10_fn>(input, stream);//✨✨✨
}

} // namespace OwnTensor