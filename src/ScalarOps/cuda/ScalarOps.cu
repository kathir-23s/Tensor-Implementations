#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"

namespace OwnTensor {
namespace { // file-local CUDA helpers & kernels

inline int half_fmt(Dtype dt) { // 0 = numeric; 1 = fp16; 2 = bf16
    return (dt == Dtype::Float16) ? 1 : (dt == Dtype::Bfloat16 ? 2 : 0);
}

__device__ inline float dev_bf16_to_float(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}
__device__ inline uint16_t dev_float_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb; // RNE
    return (uint16_t)(u >> 16);
}
__device__ inline float dev_fp16_to_float(uint16_t bits) {
    __half h = *reinterpret_cast<const __half*>(&bits);
    return __half2float(h);
}
__device__ inline uint16_t dev_float_to_fp16(float f) {
    __half h = __float2half_rn(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

template <typename T>
__device__ inline float ldf(const T* p, size_t i, int) { return static_cast<float>(p[i]); }

template <>
[[maybe_unused]] __device__ inline float ldf<uint16_t>(const uint16_t* p, size_t i, int fmt) {
    return (fmt == 1) ? dev_fp16_to_float(p[i])
         : (fmt == 2) ? dev_bf16_to_float(p[i])
                      : static_cast<float>(p[i]);
}

template <typename T>
__device__ inline void stf(T* p, size_t i, float v, int) { p[i] = static_cast<T>(v); }

template <>
[[maybe_unused]] __device__ inline void stf<uint16_t>(uint16_t* p, size_t i, float v, int fmt) {
    p[i] = (fmt == 1) ? dev_float_to_fp16(v)
         : (fmt == 2) ? dev_float_to_bf16(v)
                      : static_cast<uint16_t>(v);
}

inline dim3 pick_grid(size_t n, dim3 b) {
    size_t blocks = (n + b.x - 1) / b.x;
    if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline void ckerr(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}

// ---- kernels (in-place) ----
template<typename T>
__global__ void k_add_inplace(T* d, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(d, i, ldf<T>(d, i, fmt) + s, fmt);
}
template<typename T>
__global__ void k_sub_inplace(T* d, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(d, i, ldf<T>(d, i, fmt) - s, fmt);
}
template<typename T>
__global__ void k_mul_inplace(T* d, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(d, i, ldf<T>(d, i, fmt) * s, fmt);
}
template<typename T>
__global__ void k_div_inplace(T* d, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(d, i, ldf<T>(d, i, fmt) / s, fmt);
}

// ---- kernels (copy) ----
template<typename T>
__global__ void k_add_copy(const T* a, T* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(o, i, ldf<T>(a, i, fmt) + s, fmt);
}
template<typename T>
__global__ void k_sub_copy(const T* a, T* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(o, i, ldf<T>(a, i, fmt) - s, fmt);
}
template<typename T>
__global__ void k_mul_copy(const T* a, T* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(o, i, ldf<T>(a, i, fmt) * s, fmt);
}
template<typename T>
__global__ void k_div_copy(const T* a, T* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(o, i, ldf<T>(a, i, fmt) / s, fmt);
}

template<typename T>
__global__ void k_sub_copy_scalar_tensor(const T* a, T* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        stf<T>(o, i, s - ldf<T>(a, i, fmt), fmt);
}

template<typename T>
__global__ void k_div_copy_scalar_tensor(const T* a, T* o, float s, size_t n, int fmt, int* flag) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (std::is_integral_v<T>) {
            if (fmt == 0 && a[i] == (T)0) { if (flag) atomicExch(flag, 1); }
        }
        stf<T>(o, i, s / ldf<T>(a, i, fmt), fmt);
    }
}

// ---- launch helpers ----
template <typename T, typename Kernel>
inline void launch_copy(const Tensor& a, Tensor& out, double s, Kernel k) {
    const size_t n = a.numel();
    const dim3 block = dim3(256), grid = pick_grid(n, block);
    const int fmt = half_fmt(a.dtype());
    k<<<grid, block>>>(a.data<T>(), out.data<T>(), (float)s, n, fmt);
    ckerr("scalar copy");
}

template <typename T, typename Kernel>
inline void launch_inplace(Tensor& t, double s, Kernel k) {
    const size_t n = t.numel();
    const dim3 block = dim3(256), grid = pick_grid(n, block);
    const int fmt = half_fmt(t.dtype());
    k<<<grid, block>>>(t.data<T>(), (float)s, n, fmt);
    ckerr("scalar inplace");
}

} // anon namespace

// --------- public CUDA backend ---------
void cuda_add_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_add_inplace<T>); });
}
void cuda_sub_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_sub_inplace<T>); });
}
void cuda_mul_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_mul_inplace<T>); });
}
void cuda_div_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_div_inplace<T>); });
}

Tensor cuda_add_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_add_copy<T>); });
    return out;
}
Tensor cuda_sub_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_sub_copy<T>); });
    return out;
}
Tensor cuda_mul_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_mul_copy<T>); });
    return out;
}
Tensor cuda_div_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_div_copy<T>); });
    return out;
}

Tensor cuda_sub_copy_scalar_tensor(double s, const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_sub_copy_scalar_tensor<T>); });
    return out;
}

Tensor cuda_div_copy_scalar_tensor(double s, const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){
        using T = decltype(d);
        const size_t n = a.numel();
        const dim3 block = dim3(256), grid = pick_grid(n, block);
        const int fmt = half_fmt(a.dtype());

        int host_flag = 0;
        int* dev_flag = nullptr;
        cudaMalloc(&dev_flag, sizeof(int));
        cudaMemset(dev_flag, 0, sizeof(int));

        k_div_copy_scalar_tensor<T><<<grid, block>>>(a.data<T>(), out.data<T>(), (float)s, n, fmt, dev_flag);
        ckerr("k_div_copy_scalar_tensor");

        cudaMemcpy(&host_flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dev_flag);

        if (host_flag) throw std::runtime_error("Division by zero in scalar / integer tensor");
    });
    return out;
}

} // namespace OwnTensor
#endif // WITH_CUDA
