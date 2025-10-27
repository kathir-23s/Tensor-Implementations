#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ops/helpers/arith.hpp"
#include "core/Tensor.h"
#include "dtype/Types.h"

namespace OwnTensor {

// ============================================================================
// Helpers - Dtype Promotion Rules
// ============================================================================

inline Dtype promote_for_square(Dtype dt) {
    if (dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64) {
        return Dtype::Float64;
    }
    return dt;
}

inline Dtype promote_for_float_result(Dtype dt) {
    if (dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64) {
        return Dtype::Float32;
    }
    return dt;
}

// ============================================================================
// GPU Kernels - SQUARE
// ============================================================================

template<typename T_In, typename T_Out>
__global__ void square_kernel_gpu(const T_In* in, T_Out* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T_Out val = static_cast<T_Out>(in[idx]);
        out[idx] = val * val;
    }
}

__global__ void square_half_kernel(const __half* in, __half* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(in[idx]);
        out[idx] = __float2half(val * val);
    }
}

__global__ void square_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(in[idx]);
        out[idx] = __float2bfloat16(val * val);
    }
}

// ============================================================================
// GPU Kernels - SQRT
// ============================================================================

template<typename T_In, typename T_Out>
__global__ void sqrt_kernel_gpu(const T_In* in, T_Out* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T_Out>(sqrtf(static_cast<float>(in[idx])));
    }
}

template<>
__global__ void sqrt_kernel_gpu<double, double>(const double* in, double* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrt(in[idx]);
    }
}

__global__ void sqrt_half_kernel(const __half* in, __half* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(in[idx]);
        out[idx] = __float2half(sqrtf(val));
    }
}

__global__ void sqrt_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(in[idx]);
        out[idx] = __float2bfloat16(sqrtf(val));
    }
}

// ============================================================================
// GPU Kernels - POWER (with edge case handling)
// ============================================================================

template<typename T, typename ExpT>
__device__ inline T safe_pow_device(T base, ExpT exponent) {
    // Handle NaN
    if (isnan(base) || isnan(exponent)) {
        return nanf("");
    }
    
    // 0^0 = 1 (convention)
    if (base == T(0) && exponent == ExpT(0)) {
        return T(1);
    }
    
    // 0^(negative) = inf
    if (base == T(0) && exponent < ExpT(0)) {
        return INFINITY;
    }
    
    // 0^(positive) = 0
    if (base == T(0) && exponent > ExpT(0)) {
        return T(0);
    }
    
    // Negative base with non-integer exponent = NaN
    if (base < T(0) && floor(exponent) != exponent) {
        return nanf("");
    }
    
    return pow(base, static_cast<T>(exponent));
}

// Power kernel for all numeric types
template<typename T_In, typename T_Out, typename ExpT>
__global__ void power_kernel_gpu(const T_In* in, T_Out* out, size_t n, ExpT exponent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T_Out base = static_cast<T_Out>(in[idx]);
        out[idx] = safe_pow_device(base, static_cast<T_Out>(exponent));
    }
}

// Specialized power kernels for half and bfloat16
template<typename ExpT>
__global__ void power_half_kernel(const __half* in, __half* out, size_t n, ExpT exponent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float base = __half2float(in[idx]);
        float result = safe_pow_device(base, static_cast<float>(exponent));
        out[idx] = __float2half(result);
    }
}

template<typename ExpT>
__global__ void power_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, ExpT exponent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float base = __bfloat162float(in[idx]);
        float result = safe_pow_device(base, static_cast<float>(exponent));
        out[idx] = __float2bfloat16(result);
    }
}

// ============================================================================
// GPU Kernels - RECIPROCAL
// ============================================================================

template<typename T_In, typename T_Out>
__global__ void reciprocal_kernel_gpu(const T_In* in, T_Out* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = T_Out(1) / static_cast<T_Out>(in[idx]);
    }
}

__global__ void reciprocal_half_kernel(const __half* in, __half* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(in[idx]);
        out[idx] = __float2half(1.0f / val);
    }
}

__global__ void reciprocal_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(in[idx]);
        out[idx] = __float2bfloat16(1.0f / val);
    }
}

// ============================================================================
// GPU Kernels - NEGATION
// ============================================================================

template<typename T>
__global__ void negate_kernel_gpu(const T* in, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -in[idx];
    }
}

__global__ void negate_half_kernel(const __half* in, __half* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hneg(in[idx]);
    }
}

__global__ void negate_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(in[idx]);
        out[idx] = __float2bfloat16(-val);
    }
}

// ============================================================================
// GPU Kernels - ABSOLUTE
// ============================================================================

template<typename T>
__global__ void abs_kernel_gpu(const T* in, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = in[idx];
        out[idx] = (val < 0) ? -val : val;
    }
}

template<>
__global__ void abs_kernel_gpu<float>(const float* in, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(in[idx]);
    }
}

template<>
__global__ void abs_kernel_gpu<double>(const double* in, double* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabs(in[idx]);
    }
}

__global__ void abs_half_kernel(const __half* in, __half* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __habs(in[idx]);
    }
}

__global__ void abs_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(in[idx]);
        out[idx] = __float2bfloat16(fabsf(val));
    }
}

// ============================================================================
// GPU Kernels - SIGN
// ============================================================================

template<typename T>
__global__ void sign_kernel_gpu(const T* in, T* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = in[idx];
        out[idx] = (val > 0) ? T(1) : ((val < 0) ? T(-1) : T(0));
    }
}

__global__ void sign_half_kernel(const __half* in, __half* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(in[idx]);
        float result = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        out[idx] = __float2half(result);
    }
}

__global__ void sign_bfloat16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(in[idx]);
        float result = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        out[idx] = __float2bfloat16(result);
    }
}

// ============================================================================
// Helper: Launch kernel with proper grid/block configuration
// ============================================================================

template<typename Kernel, typename... Args>
void launch_kernel(Kernel kernel, size_t n, Args... args) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(args..., n);
    cudaDeviceSynchronize();
}

// ============================================================================
// Generic GPU dispatcher - dtype-aware kernel selection
// ============================================================================

template<typename KernelInt16, typename KernelInt32, typename KernelInt64,
         typename KernelFloat32, typename KernelFloat64,
         typename KernelHalf, typename KernelBFloat16>
void dispatch_gpu_kernel(Dtype dtype, const void* in, void* out, size_t n,
                        KernelInt16 k_i16, KernelInt32 k_i32, KernelInt64 k_i64,
                        KernelFloat32 k_f32, KernelFloat64 k_f64,
                        KernelHalf k_half, KernelBFloat16 k_bf16) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    switch(dtype) {
        case Dtype::Int16:
            k_i16<<<blocks, threads>>>((const int16_t*)in, (int16_t*)out, n);
            break;
        case Dtype::Int32:
            k_i32<<<blocks, threads>>>((const int32_t*)in, (int32_t*)out, n);
            break;
        case Dtype::Int64:
            k_i64<<<blocks, threads>>>((const int64_t*)in, (int64_t*)out, n);
            break;
        case Dtype::Float32:
            k_f32<<<blocks, threads>>>((const float*)in, (float*)out, n);
            break;
        case Dtype::Float64:
            k_f64<<<blocks, threads>>>((const double*)in, (double*)out, n);
            break;
        case Dtype::Float16:
            k_half<<<blocks, threads>>>((const __half*)in, (__half*)out, n);
            break;
        case Dtype::Bfloat16:
            k_bf16<<<blocks, threads>>>((const __nv_bfloat16*)in, (__nv_bfloat16*)out, n);
            break;
    }
    cudaDeviceSynchronize();
}

// ============================================================================
// SQUARE - GPU Wrappers
// ============================================================================
Tensor square_out_gpu_wrap(const Tensor& input) {
    Dtype out_dtype = promote_for_square(input.dtype());
    Tensor output(input.shape(), out_dtype, input.device(), input.requires_grad());
    size_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Handle special cases
    if (input.dtype() == Dtype::Float16) {
        square_half_kernel<<<blocks, threads>>>(input.data<__half>(), output.data<__half>(), n);
    } else if (input.dtype() == Dtype::Bfloat16) {
        square_bfloat16_kernel<<<blocks, threads>>>(input.data<__nv_bfloat16>(), output.data<__nv_bfloat16>(), n);
    } else if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        // Integers promoted to Float64
        if (input.dtype() == Dtype::Int16) {
            square_kernel_gpu<<<blocks, threads>>>(input.data<int16_t>(), output.data<double>(), n);
        } else if (input.dtype() == Dtype::Int32) {
            square_kernel_gpu<<<blocks, threads>>>(input.data<int32_t>(), output.data<double>(), n);
        } else {
            square_kernel_gpu<<<blocks, threads>>>(input.data<int64_t>(), output.data<double>(), n);
        }
    } else if (input.dtype() == Dtype::Float32) {
        square_kernel_gpu<<<blocks, threads>>>(input.data<float>(), output.data<float>(), n);
    } else if (input.dtype() == Dtype::Float64) {
        square_kernel_gpu<<<blocks, threads>>>(input.data<double>(), output.data<double>(), n);
    }
    
    cudaDeviceSynchronize();
    return output;
}

void square_in_gpu_wrap(Tensor& input) {
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), input.data(), n,
        square_kernel_gpu<int16_t, int16_t>, square_kernel_gpu<int32_t, int32_t>,
        square_kernel_gpu<int64_t, int64_t>, square_kernel_gpu<float, float>,
        square_kernel_gpu<double, double>, square_half_kernel, square_bfloat16_kernel);
}

// ============================================================================
// SQUARE ROOT - GPU Wrappers
// ============================================================================

Tensor square_root_out_gpu_wrap(const Tensor& input) {
    Dtype out_dtype = promote_for_float_result(input.dtype());
    Tensor output(input.shape(), out_dtype, input.device(), input.requires_grad());
    size_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    if (input.dtype() == Dtype::Float16) {
        sqrt_half_kernel<<<blocks, threads>>>(input.data<__half>(), output.data<__half>(), n);
    } else if (input.dtype() == Dtype::Bfloat16) {
        sqrt_bfloat16_kernel<<<blocks, threads>>>(input.data<__nv_bfloat16>(), output.data<__nv_bfloat16>(), n);
    } else if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        // Integers promoted to Float32
        if (input.dtype() == Dtype::Int16) {
            sqrt_kernel_gpu<<<blocks, threads>>>(input.data<int16_t>(), output.data<float>(), n);
        } else if (input.dtype() == Dtype::Int32) {
            sqrt_kernel_gpu<<<blocks, threads>>>(input.data<int32_t>(), output.data<float>(), n);
        } else {
            sqrt_kernel_gpu<<<blocks, threads>>>(input.data<int64_t>(), output.data<float>(), n);
        }
    } else if (input.dtype() == Dtype::Float32) {
        sqrt_kernel_gpu<<<blocks, threads>>>(input.data<float>(), output.data<float>(), n);
    } else if (input.dtype() == Dtype::Float64) {
        sqrt_kernel_gpu<<<blocks, threads>>>(input.data<double>(), output.data<double>(), n);
    }
    
    cudaDeviceSynchronize();
    return output;
}

void square_root_in_gpu_wrap(Tensor& input) {
    if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        throw std::invalid_argument("In-place square root not supported for integer tensors");
    }
    
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), input.data(), n,
        sqrt_kernel_gpu<int16_t, int16_t>, sqrt_kernel_gpu<int32_t, int32_t>,
        sqrt_kernel_gpu<int64_t, int64_t>, sqrt_kernel_gpu<float, float>,
        sqrt_kernel_gpu<double, double>, sqrt_half_kernel, sqrt_bfloat16_kernel);
}

// ============================================================================
// POWER - GPU Wrappers (int, float, double exponents)
// ============================================================================
// Helper template to launch power kernel for a specific exponent type
template<typename ExpT>
Tensor power_out_gpu_wrap_impl(const Tensor& input, ExpT exponent) {
    Dtype out_dtype = promote_for_float_result(input.dtype());
    Tensor output(input.shape(), out_dtype, input.device(), input.requires_grad());
    size_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (input.dtype() == Dtype::Float16) {
        power_half_kernel<<<blocks, threads>>>(input.data<__half>(), output.data<__half>(), n, exponent);
    } else if (input.dtype() == Dtype::Bfloat16) {
        power_bfloat16_kernel<<<blocks, threads>>>(input.data<__nv_bfloat16>(), output.data<__nv_bfloat16>(), n, exponent);
    } else if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        if (input.dtype() == Dtype::Int16) {
            power_kernel_gpu<<<blocks, threads>>>(input.data<int16_t>(), output.data<float>(), n, exponent);
        } else if (input.dtype() == Dtype::Int32) {
            power_kernel_gpu<<<blocks, threads>>>(input.data<int32_t>(), output.data<float>(), n, exponent);
        } else {
            power_kernel_gpu<<<blocks, threads>>>(input.data<int64_t>(), output.data<float>(), n, exponent);
        }
    } else if (input.dtype() == Dtype::Float32) {
        power_kernel_gpu<<<blocks, threads>>>(input.data<float>(), output.data<float>(), n, exponent);
    } else if (input.dtype() == Dtype::Float64) {
        power_kernel_gpu<<<blocks, threads>>>(input.data<double>(), output.data<double>(), n, exponent);
    }
    
    cudaDeviceSynchronize();
    return output;
}

template<typename ExpT>
void power_in_gpu_wrap_impl(Tensor& input, ExpT exponent) {
    if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        throw std::invalid_argument("In-place power not supported for integer tensors");
    }
    
    size_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (input.dtype() == Dtype::Float16) {
        power_half_kernel<<<blocks, threads>>>(input.data<__half>(), input.data<__half>(), n, exponent);
    } else if (input.dtype() == Dtype::Bfloat16) {
        power_bfloat16_kernel<<<blocks, threads>>>(input.data<__nv_bfloat16>(), input.data<__nv_bfloat16>(), n, exponent);
    } else if (input.dtype() == Dtype::Float32) {
        power_kernel_gpu<<<blocks, threads>>>(input.data<float>(), input.data<float>(), n, exponent);
    } else if (input.dtype() == Dtype::Float64) {
        power_kernel_gpu<<<blocks, threads>>>(input.data<double>(), input.data<double>(), n, exponent);
    }
    
    cudaDeviceSynchronize();
}

// Public wrappers for different exponent types
Tensor power_out_gpu_wrap(const Tensor& input, int exponent) {
    return power_out_gpu_wrap_impl(input, exponent);
}

Tensor power_out_gpu_wrap(const Tensor& input, float exponent) {
    return power_out_gpu_wrap_impl(input, exponent);
}

Tensor power_out_gpu_wrap(const Tensor& input, double exponent) {
    return power_out_gpu_wrap_impl(input, exponent);
}

void power_in_gpu_wrap(Tensor& input, int exponent) {
    power_in_gpu_wrap_impl(input, exponent);
}

void power_in_gpu_wrap(Tensor& input, float exponent) {
    power_in_gpu_wrap_impl(input, exponent);
}

void power_in_gpu_wrap(Tensor& input, double exponent) {
    power_in_gpu_wrap_impl(input, exponent);
}

// ============================================================================
// RECIPROCAL - GPU Wrappers
// ============================================================================

Tensor reciprocal_out_gpu_wrap(const Tensor& input) {
    Dtype out_dtype = promote_for_float_result(input.dtype());
    Tensor output(input.shape(), out_dtype, input.device(), input.requires_grad());
    size_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    if (input.dtype() == Dtype::Float16) {
        reciprocal_half_kernel<<<blocks, threads>>>(input.data<__half>(), output.data<__half>(), n);
    } else if (input.dtype() == Dtype::Bfloat16) {
        reciprocal_bfloat16_kernel<<<blocks, threads>>>(input.data<__nv_bfloat16>(), output.data<__nv_bfloat16>(), n);
    } else if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        if (input.dtype() == Dtype::Int16) {
            reciprocal_kernel_gpu<<<blocks, threads>>>(input.data<int16_t>(), output.data<float>(), n);
        } else if (input.dtype() == Dtype::Int32) {
            reciprocal_kernel_gpu<<<blocks, threads>>>(input.data<int32_t>(), output.data<float>(), n);
        } else {
            reciprocal_kernel_gpu<<<blocks, threads>>>(input.data<int64_t>(), output.data<float>(), n);
        }
    } else if (input.dtype() == Dtype::Float32) {
        reciprocal_kernel_gpu<<<blocks, threads>>>(input.data<float>(), output.data<float>(), n);
    } else if (input.dtype() == Dtype::Float64) {
        reciprocal_kernel_gpu<<<blocks, threads>>>(input.data<double>(), output.data<double>(), n);
    }
    
    cudaDeviceSynchronize();
    return output;
}

void reciprocal_in_gpu_wrap(Tensor& input) {
    if (input.dtype() == Dtype::Int16 || input.dtype() == Dtype::Int32 || input.dtype() == Dtype::Int64) {
        throw std::invalid_argument("In-place reciprocal not supported for integer tensors");
    }
    
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), input.data(), n,
        reciprocal_kernel_gpu<int16_t, int16_t>, reciprocal_kernel_gpu<int32_t, int32_t>,
        reciprocal_kernel_gpu<int64_t, int64_t>, reciprocal_kernel_gpu<float, float>,
        reciprocal_kernel_gpu<double, double>, reciprocal_half_kernel, reciprocal_bfloat16_kernel);
}

// ============================================================================
// NEGATION - GPU Wrappers
// ============================================================================

Tensor negator_out_gpu_wrap(const Tensor& input) {
    Tensor output(input.shape(), input.dtype(), input.device(), input.requires_grad());
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), output.data(), n,
        negate_kernel_gpu<int16_t>, negate_kernel_gpu<int32_t>, negate_kernel_gpu<int64_t>,
        negate_kernel_gpu<float>, negate_kernel_gpu<double>,
        negate_half_kernel, negate_bfloat16_kernel);
    return output;
}

void negator_in_gpu_wrap(Tensor& input) {
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), input.data(), n,
        negate_kernel_gpu<int16_t>, negate_kernel_gpu<int32_t>, negate_kernel_gpu<int64_t>,
        negate_kernel_gpu<float>, negate_kernel_gpu<double>,
        negate_half_kernel, negate_bfloat16_kernel);
}

// ============================================================================
// ABSOLUTE - GPU Wrappers
// ============================================================================

Tensor absolute_out_gpu_wrap(const Tensor& input) {
    Tensor output(input.shape(), input.dtype(), input.device(), input.requires_grad());
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), output.data(), n,
        abs_kernel_gpu<int16_t>, abs_kernel_gpu<int32_t>, abs_kernel_gpu<int64_t>,
        abs_kernel_gpu<float>, abs_kernel_gpu<double>,
        abs_half_kernel, abs_bfloat16_kernel);
    return output;
}

void absolute_in_gpu_wrap(Tensor& input) {
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), input.data(), n,
        abs_kernel_gpu<int16_t>, abs_kernel_gpu<int32_t>, abs_kernel_gpu<int64_t>,
        abs_kernel_gpu<float>, abs_kernel_gpu<double>,
        abs_half_kernel, abs_bfloat16_kernel);
}

// ============================================================================
// SIGN - GPU Wrappers
// ============================================================================

Tensor sign_out_gpu_wrap(const Tensor& input) {
    Tensor output(input.shape(), input.dtype(), input.device(), input.requires_grad());
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), output.data(), n,
        sign_kernel_gpu<int16_t>, sign_kernel_gpu<int32_t>, sign_kernel_gpu<int64_t>,
        sign_kernel_gpu<float>, sign_kernel_gpu<double>,
        sign_half_kernel, sign_bfloat16_kernel);
    return output;
}

void sign_in_gpu_wrap(Tensor& input) {
    size_t n = input.numel();
    dispatch_gpu_kernel(input.dtype(), input.data(), input.data(), n,
        sign_kernel_gpu<int16_t>, sign_kernel_gpu<int32_t>, sign_kernel_gpu<int64_t>,
        sign_kernel_gpu<float>, sign_kernel_gpu<double>,
        sign_half_kernel, sign_bfloat16_kernel);
}


} // namespace OwnTensor