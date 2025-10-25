#ifdef WITH_CUDA

#include "ops/ScalarOps.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"  
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <type_traits>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {

// ======================================================================
// Small utilities (launch + error check)
// ======================================================================

inline dim3 pick_block(size_t /*n*/) { return dim3(256); }
inline dim3 pick_grid(size_t n, dim3 block) {
    size_t blocks = (n + block.x - 1) / block.x;
    if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline void throw_if_cuda_error(const char* where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

// ======================================================================
// Conversions for uint16_t-backed half types on device
// fmt: 0 = normal numeric types; 1 = Float16; 2 = Bfloat16
// ======================================================================

__device__ inline float dev_bf16_to_float(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}
__device__ inline uint16_t dev_float_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    // If you want RNE on device: uint32_t lsb=(u>>16)&1u; u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

__device__ inline float dev_fp16_to_float(uint16_t hbits) {
    __half h = *reinterpret_cast<const __half*>(&hbits);
    return __half2float(h);
}
__device__ inline uint16_t dev_float_to_fp16(float f) {
    __half h = __float2half_rn(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

// typed load/store that consider fmt only for uint16_t storage
template <typename T>
__device__ inline float dev_load_as_float(const T* p, size_t i, int /*fmt*/) {
    return (float)p[i];
}
template <>
__device__ inline float dev_load_as_float<uint16_t>(const uint16_t* p, size_t i, int fmt) {
    return (fmt == 1) ? dev_fp16_to_float(p[i])
         : (fmt == 2) ? dev_bf16_to_float(p[i])
                      : (float)p[i];
}

template <typename T>
__device__ inline void dev_store_from_float(T* p, size_t i, float v, int /*fmt*/) {
    p[i] = (T)v;
}
template <>
__device__ inline void dev_store_from_float<uint16_t>(uint16_t* p, size_t i, float v, int fmt) {
    p[i] = (fmt == 1) ? dev_float_to_fp16(v)
         : (fmt == 2) ? dev_float_to_bf16(v)
                      : (uint16_t)v;
}

// ======================================================================
// Kernels (scalar passed as float to avoid host-only conversions on device)
// ======================================================================

template<typename T>
__global__ void k_add_inplace(T* data, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(data, i, fmt);
        dev_store_from_float<T>(data, i, v + s, fmt);
    }
}

template<typename T>
__global__ void k_sub_inplace(T* data, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(data, i, fmt);
        dev_store_from_float<T>(data, i, v - s, fmt);
    }
}

template<typename T>
__global__ void k_mul_inplace(T* data, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(data, i, fmt);
        dev_store_from_float<T>(data, i, v * s, fmt);
    }
}

template<typename T>
__global__ void k_div_inplace(T* data, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(data, i, fmt);
        dev_store_from_float<T>(data, i, v / s, fmt);
    }
}

template<typename T>
__global__ void k_add_copy(const T* src, T* dst, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(src, i, fmt);
        dev_store_from_float<T>(dst, i, v + s, fmt);
    }
}

template<typename T>
__global__ void k_sub_copy(const T* src, T* dst, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(src, i, fmt);
        dev_store_from_float<T>(dst, i, v - s, fmt);
    }
}

template<typename T>
__global__ void k_mul_copy(const T* src, T* dst, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(src, i, fmt);
        dev_store_from_float<T>(dst, i, v * s, fmt);
    }
}

template<typename T>
__global__ void k_div_copy(const T* src, T* dst, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(src, i, fmt);
        dev_store_from_float<T>(dst, i, v / s, fmt);
    }
}

// scalar MINUS tensor (dst = s - src)
template<typename T>
__global__ void k_sub_copy_scalar_tensor(const T* src, T* dst, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = dev_load_as_float<T>(src, i, fmt);
        dev_store_from_float<T>(dst, i, s - v, fmt);
    }
}

// scalar DIV tensor (dst = s / src); integer tensors: flag divide-by-zero
template<typename T>
__global__ void k_div_copy_scalar_tensor(const T* src, T* dst, float s, size_t n, int fmt, int* error_flag) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if constexpr (std::is_integral_v<T>) {
            if (src[i] == (T)0) { if (error_flag) atomicExch(error_flag, 1); }
        }
        const float v = dev_load_as_float<T>(src, i, fmt);
        dev_store_from_float<T>(dst, i, s / v, fmt);
    }
}

// ======================================================================
// Public operator templates (GPU path)
// ======================================================================

template<typename S>
Tensor& operator+=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    cudaStream_t stream = nullptr; // or tensor.device().cuda_stream();
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar); // host-side cast OK

        k_add_inplace<T><<<grid, block, 0, stream>>>(data, s, n, fmt);
        throw_if_cuda_error("k_add_inplace");
    });
    return tensor;
}

template<typename S>
Tensor& operator-=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_sub_inplace<T><<<grid, block, 0, stream>>>(data, s, n, fmt);
        throw_if_cuda_error("k_sub_inplace");
    });
    return tensor;
}

template<typename S>
Tensor& operator*=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_mul_inplace<T><<<grid, block, 0, stream>>>(data, s, n, fmt);
        throw_if_cuda_error("k_mul_inplace");
    });
    return tensor;
}

template<typename S>
Tensor& operator/=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T>) {
            if ((double)static_cast<float>(scalar) == 0.0) {
                throw std::runtime_error("Division by zero in integer tensor /=");
            }
        }
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_div_inplace<T><<<grid, block, 0, stream>>>(data, s, n, fmt);
        throw_if_cuda_error("k_div_inplace");
    });
    return tensor;
}

template<typename S>
Tensor operator+(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_add_copy<T><<<grid, block, 0, stream>>>(src, dst, s, n, fmt);
        throw_if_cuda_error("k_add_copy");
    });
    return out;
}

template<typename S>
Tensor operator-(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_sub_copy<T><<<grid, block, 0, stream>>>(src, dst, s, n, fmt);
        throw_if_cuda_error("k_sub_copy");
    });
    return out;
}

template<typename S>
Tensor operator*(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_mul_copy<T><<<grid, block, 0, stream>>>(src, dst, s, n, fmt);
        throw_if_cuda_error("k_mul_copy");
    });
    return out;
}

template<typename S>
Tensor operator/(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T>) {
            if ((double)static_cast<float>(scalar) == 0.0) {
                throw std::runtime_error("Division by zero in integer tensor / scalar");
            }
        }
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_div_copy<T><<<grid, block, 0, stream>>>(src, dst, s, n, fmt);
        throw_if_cuda_error("k_div_copy");
    });
    return out;
}

template<typename S>
Tensor operator+(S scalar, const Tensor& tensor) {
    return tensor + scalar;
}

template<typename S>
Tensor operator-(S scalar, const Tensor& tensor) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    cudaStream_t stream = nullptr;
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_sub_copy_scalar_tensor<T><<<grid, block, 0, stream>>>(src, dst, s, n, fmt);
        throw_if_cuda_error("k_sub_copy_scalar_tensor");
    });
    return out;
}

template<typename S>
Tensor operator*(S scalar, const Tensor& tensor) {
    return tensor * scalar;
}

template<typename S>
Tensor operator/(S scalar, const Tensor& tensor) {
    if (!tensor.device().is_cuda())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    cudaStream_t stream = nullptr;

    int h_flag = 0;
    int* d_flag = nullptr;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemsetAsync(d_flag, 0, sizeof(int), stream);

    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const dim3 block = pick_block(n), grid = pick_grid(n, block);
        const int fmt = (tensor.dtype() == Dtype::Float16) ? 1
                   : (tensor.dtype() == Dtype::Bfloat16) ? 2 : 0;
        const float s = static_cast<float>(scalar);

        k_div_copy_scalar_tensor<T><<<grid, block, 0, stream>>>(src, dst, s, n, fmt, d_flag);
        throw_if_cuda_error("k_div_copy_scalar_tensor");
    });

    cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_flag);

    if (h_flag) throw std::runtime_error("Division by zero in scalar / integer tensor");
    return out;
}

// ======================================================================
// Explicit instantiations (GPU TU)
// ======================================================================

template Tensor& operator+=<int16_t>(Tensor&, int16_t);
template Tensor& operator+=<int32_t>(Tensor&, int32_t);
template Tensor& operator+=<int64_t>(Tensor&, int64_t);
template Tensor& operator+=<float>(Tensor&, float);
template Tensor& operator+=<double>(Tensor&, double);
template Tensor& operator+=<float16_t>(Tensor&, float16_t);
template Tensor& operator+=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator-=<int16_t>(Tensor&, int16_t);
template Tensor& operator-=<int32_t>(Tensor&, int32_t);
template Tensor& operator-=<int64_t>(Tensor&, int64_t);
template Tensor& operator-=<float>(Tensor&, float);
template Tensor& operator-=<double>(Tensor&, double);
template Tensor& operator-=<float16_t>(Tensor&, float16_t);
template Tensor& operator-=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator*=<int16_t>(Tensor&, int16_t);
template Tensor& operator*=<int32_t>(Tensor&, int32_t);
template Tensor& operator*=<int64_t>(Tensor&, int64_t);
template Tensor& operator*=<float>(Tensor&, float);
template Tensor& operator*=<double>(Tensor&, double);
template Tensor& operator*=<float16_t>(Tensor&, float16_t);
template Tensor& operator*=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator/=<int16_t>(Tensor&, int16_t);
template Tensor& operator/=<int32_t>(Tensor&, int32_t);
template Tensor& operator/=<int64_t>(Tensor&, int64_t);
template Tensor& operator/=<float>(Tensor&, float);
template Tensor& operator/=<double>(Tensor&, double);
template Tensor& operator/=<float16_t>(Tensor&, float16_t);
template Tensor& operator/=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor operator+<int16_t>(const Tensor&, int16_t);
template Tensor operator+<int32_t>(const Tensor&, int32_t);
template Tensor operator+<int64_t>(const Tensor&, int64_t);
template Tensor operator+<float>(const Tensor&, float);
template Tensor operator+<double>(const Tensor&, double);
template Tensor operator+<float16_t>(const Tensor&, float16_t);
template Tensor operator+<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator-<int16_t>(const Tensor&, int16_t);
template Tensor operator-<int32_t>(const Tensor&, int32_t);
template Tensor operator-<int64_t>(const Tensor&, int64_t);
template Tensor operator-<float>(const Tensor&, float);
template Tensor operator-<double>(const Tensor&, double);
template Tensor operator-<float16_t>(const Tensor&, float16_t);
template Tensor operator-<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator*<int16_t>(const Tensor&, int16_t);
template Tensor operator*<int32_t>(const Tensor&, int32_t);
template Tensor operator*<int64_t>(const Tensor&, int64_t);
template Tensor operator*<float>(const Tensor&, float);
template Tensor operator*<double>(const Tensor&, double);
template Tensor operator*<float16_t>(const Tensor&, float16_t);
template Tensor operator*<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator/<int16_t>(const Tensor&, int16_t);
template Tensor operator/<int32_t>(const Tensor&, int32_t);
template Tensor operator/<int64_t>(const Tensor&, int64_t);
template Tensor operator/<float>(const Tensor&, float);
template Tensor operator/<double>(const Tensor&, double);
template Tensor operator/<float16_t>(const Tensor&, float16_t);
template Tensor operator/<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator+<int16_t>(int16_t, const Tensor&);
template Tensor operator+<int32_t>(int32_t, const Tensor&);
template Tensor operator+<int64_t>(int64_t, const Tensor&);
template Tensor operator+<float>(float, const Tensor&);
template Tensor operator+<double>(double, const Tensor&);
template Tensor operator+<float16_t>(float16_t, const Tensor&);
template Tensor operator+<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator-<int16_t>(int16_t, const Tensor&);
template Tensor operator-<int32_t>(int32_t, const Tensor&);
template Tensor operator-<int64_t>(int64_t, const Tensor&);
template Tensor operator-<float>(float, const Tensor&);
template Tensor operator-<double>(double, const Tensor&);
template Tensor operator-<float16_t>(float16_t, const Tensor&);
template Tensor operator-<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator*<int16_t>(int16_t, const Tensor&);
template Tensor operator*<int32_t>(int32_t, const Tensor&);
template Tensor operator*<int64_t>(int64_t, const Tensor&);
template Tensor operator*<float>(float, const Tensor&);
template Tensor operator*<double>(double, const Tensor&);
template Tensor operator*<float16_t>(float16_t, const Tensor&);
template Tensor operator*<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator/<int16_t>(int16_t, const Tensor&);
template Tensor operator/<int32_t>(int32_t, const Tensor&);
template Tensor operator/<int64_t>(int64_t, const Tensor&);
template Tensor operator/<float>(float, const Tensor&);
template Tensor operator/<double>(double, const Tensor&);
template Tensor operator/<float16_t>(float16_t, const Tensor&);
template Tensor operator/<bfloat16_t>(bfloat16_t, const Tensor&);

} // namespace OwnTensor

#endif // WITH_CUDA
