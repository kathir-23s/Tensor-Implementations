#include <iostream>
#include <stdexcept>
#include <cmath>
#include "../include/ops/helpers/Trigonometry.hpp"
#include "core/Tensor.h"
#include "dtype/Types.h"

// For f16 and bf16
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {

// ============================================================================
// Device Function Pointers for GPU Trigonometric Operations
// ============================================================================
static inline __device__ float sinf_fn(float x) { return sinf(x); }
static inline __device__ double sin_fn(double x) { return sin(x); }
static inline __device__ float cosf_fn(float x) { return cosf(x); }
static inline __device__ double cos_fn(double x) { return cos(x); }
static inline __device__ float tanf_fn(float x) { return tanf(x); }
static inline __device__ double tan_fn(double x) { return tan(x); }

static inline __device__ float asinf_fn(float x) { return asinf(x); }
static inline __device__ double asin_fn(double x) { return asin(x); }
static inline __device__ float acosf_fn(float x) { return acosf(x); }
static inline __device__ double acos_fn(double x) { return acos(x); }
static inline __device__ float atanf_fn(float x) { return atanf(x); }
static inline __device__ double atan_fn(double x) { return atan(x); }

static inline __device__ float sinhf_fn(float x) { return sinhf(x); }
static inline __device__ double sinh_fn(double x) { return sinh(x); }
static inline __device__ float coshf_fn(float x) { return coshf(x); }
static inline __device__ double cosh_fn(double x) { return cosh(x); }
static inline __device__ float tanhf_fn(float x) { return tanhf(x); }
static inline __device__ double tanh_fn(double x) { return tanh(x); }

static inline __device__ float asinhf_fn(float x) { return asinhf(x); }
static inline __device__ double asinh_fn(double x) { return asinh(x); }
static inline __device__ float acoshf_fn(float x) { return acoshf(x); }
static inline __device__ double acosh_fn(double x) { return acosh(x); }
static inline __device__ float atanhf_fn(float x) { return atanhf(x); }
static inline __device__ double atanh_fn(double x) { return atanh(x); }

// ============================================================================
// Generic CUDA Unary Kernel (for standard types) - REUSE FROM EXPONENTS
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
// Specialized CUDA Kernel for Float16 (half precision) - REUSE FROM EXPONENTS
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
// Specialized CUDA Kernel for Bfloat16 - REUSE FROM EXPONENTS
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
// Helper: Determine output dtype for integer promotion - REUSE FROM EXPONENTS
// ============================================================================
inline Dtype get_promoted_dtype_trig(Dtype input_dtype) {
    switch(input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Bool:
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
static auto dispatch_gpu_dtype_trig(Dtype dtype, Func&& f) {
    switch(dtype) {
        // Integer types
        case Dtype::Int16: return f(int16_t{});
        case Dtype::Int32: return f(int32_t{});
        case Dtype::Int64: return f(int64_t{});
        case Dtype::Bool:return f(uint8_t{});
        // Floating point types
        case Dtype::Float32: return f(float{});
        case Dtype::Float64: return f(double{});
        // Explicitly exclude bf16/f16 - they should be handled separately
        case Dtype::Float16:
        case Dtype::Bfloat16:
            throw std::runtime_error("Float16/Bfloat16 should be handled before dispatch!");
        default:
            throw std::runtime_error("Unsupported dtype in dispatch_gpu_dtype_trig");
    }
}

// ============================================================================
// Generic Out-of-Place GPU Wrapper for Trigonometric Functions
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
Tensor generic_trigonometric_out_gpu(const Tensor& input_tensor, cudaStream_t stream) {//✨✨✨
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
        // cudaDeviceSynchronize();//✨✨✨
        return output_tensor;
    }
    
    // Handle Bfloat16 - uses CUDA native bfloat16 type
    if (in_dtype == Dtype::Bfloat16) {
        Tensor output_tensor(input_tensor.shape(), in_dtype, 
                           input_tensor.device(), input_tensor.requires_grad());
        const __nv_bfloat16* in = input_tensor.data<__nv_bfloat16>();
        __nv_bfloat16* out = output_tensor.data<__nv_bfloat16>();
        
        unary_bfloat16_kernel_gpu<FloatFunc><<<blocks, threads, 0, stream>>>(in, out, size);//✨✨✨
        // cudaDeviceSynchronize();//✨✨✨
        return output_tensor;
    }
    
    // Now dispatch only handles int/float/double types
    Dtype output_dtype = get_promoted_dtype_trig(in_dtype);
    Tensor output_tensor(input_tensor.shape(), output_dtype, 
                        input_tensor.device(), input_tensor.requires_grad());
    
    // Use GPU-specific dispatch for input type
    dispatch_gpu_dtype_trig(in_dtype, [&](auto in_type_instance) {
        using InputType = decltype(in_type_instance);
        
        // Use GPU-specific dispatch for output type
        dispatch_gpu_dtype_trig(output_dtype, [&](auto out_type_instance) {
            using OutputType = decltype(out_type_instance);
            
            const InputType* in_ptr = input_tensor.data<InputType>();
            OutputType* out_ptr = output_tensor.data<OutputType>();
            
            // Select appropriate function based on output type
            if constexpr (std::is_same_v<OutputType, float>) {
                unary_kernel_gpu<InputType, OutputType, FloatFunc><<<blocks, threads, 0, stream>>>(
                    in_ptr, out_ptr, size
                );//✨✨✨
            } else if constexpr (std::is_same_v<OutputType, double>) {
                unary_kernel_gpu<InputType, OutputType, DoubleFunc><<<blocks, threads, 0, stream>>>(
                    in_ptr, out_ptr, size
                );//✨✨✨
            }
        });
    });
    
    // cudaDeviceSynchronize();//✨✨✨
    return output_tensor;
}

// ============================================================================
// Generic In-Place GPU Wrapper for Trigonometric Functions
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
void generic_trigonometric_in_gpu(Tensor& input_tensor, cudaStream_t stream) {//✨✨✨
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
        // cudaDeviceSynchronize();//✨✨✨
        return;
    }
    
    // Handle Bfloat16
    if (dtype == Dtype::Bfloat16) {
        __nv_bfloat16* data_ptr = input_tensor.data<__nv_bfloat16>();
        unary_bfloat16_kernel_gpu<FloatFunc><<<blocks, threads, 0, stream>>>(data_ptr, data_ptr, size);//✨✨✨
        // cudaDeviceSynchronize();//✨✨✨
        return;
    }
    
    // Now dispatch only handles float/double types
    dispatch_gpu_dtype_trig(dtype, [&](auto type_instance) {
        using DataType = decltype(type_instance);
        
        DataType* data_ptr = input_tensor.data<DataType>();
        
        if constexpr (std::is_same_v<DataType, float>) {
            unary_kernel_gpu<DataType, DataType, FloatFunc><<<blocks, threads, 0, stream>>>(
                data_ptr, data_ptr, size
            );//✨✨✨
        } else if constexpr (std::is_same_v<DataType, double>) {
            unary_kernel_gpu<DataType, DataType, DoubleFunc><<<blocks, threads, 0, stream>>>(
                data_ptr, data_ptr, size
            );//✨✨✨
        }
    });
    
    ////✨✨✨
     // Wait for GPU to finish
    // cudaError_t err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
    // }
    //✨✨✨
}

// ============================================================================
// GPU Wrapper Functions - Trigonometric Operations
// ============================================================================

//✨✨✨
// Basic trigonometric functions
Tensor sin_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<sinf_fn, sin_fn>(input, stream);
}

void sin_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<sinf_fn, sin_fn>(input, stream);
}

Tensor cos_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<cosf_fn, cos_fn>(input, stream);
}

void cos_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<cosf_fn, cos_fn>(input, stream);
}

Tensor tan_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<tanf_fn, tan_fn>(input, stream);
}

void tan_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<tanf_fn, tan_fn>(input, stream);
}

// Inverse trigonometric functions
Tensor asin_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<asinf_fn, asin_fn>(input, stream);
}

void asin_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<asinf_fn, asin_fn>(input, stream);
}

Tensor acos_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<acosf_fn, acos_fn>(input, stream);
}

void acos_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<acosf_fn, acos_fn>(input, stream);
}

Tensor atan_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<atanf_fn, atan_fn>(input, stream);
}

void atan_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<atanf_fn, atan_fn>(input, stream);
}

// Hyperbolic functions
Tensor sinh_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<sinhf_fn, sinh_fn>(input, stream);
}

void sinh_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<sinhf_fn, sinh_fn>(input, stream);
}

Tensor cosh_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<coshf_fn, cosh_fn>(input, stream);
}

void cosh_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<coshf_fn, cosh_fn>(input, stream);
}

Tensor tanh_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<tanhf_fn, tanh_fn>(input, stream);
}

void tanh_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<tanhf_fn, tanh_fn>(input, stream);
}

// Inverse hyperbolic functions
Tensor asinh_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<asinhf_fn, asinh_fn>(input, stream);
}

void asinh_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<asinhf_fn, asinh_fn>(input, stream);
}

Tensor acosh_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<acoshf_fn, acosh_fn>(input, stream);
}

void acosh_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<acoshf_fn, acosh_fn>(input, stream);
}

Tensor atanh_out_gpu_wrap(const Tensor& input, cudaStream_t stream) {
    return generic_trigonometric_out_gpu<atanhf_fn, atanh_fn>(input, stream);
}

void atanh_in_gpu_wrap(Tensor& input, cudaStream_t stream) {
    generic_trigonometric_in_gpu<atanhf_fn, atanh_fn>(input, stream);
}
//✨✨✨

} // namespace OwnTensor