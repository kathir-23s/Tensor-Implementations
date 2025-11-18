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
#include "dtype/DtypeTraits.h"  // ✅ For get_division_output_dtype

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

// ✅ Helper to determine promoted dtype for division (HOST function)
inline Dtype get_division_output_dtype(Dtype input_dtype) {
    if (input_dtype == Dtype::Bool) return Dtype::Float32;
    if (input_dtype == Dtype::Int16 || input_dtype == Dtype::Int32 || input_dtype == Dtype::Int64) {
        return Dtype::Float32;
    }
    return input_dtype;  // Float types stay the same
}

// ============================================================================
// ARITHMETIC KERNELS (In-place)
// ============================================================================
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

// ============================================================================
// ARITHMETIC KERNELS (Copy - same type)
// ============================================================================
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

// ✅ NEW: Cross-type division kernel (SrcT → DstT)
template<typename SrcT, typename DstT>
__global__ void k_div_copy_cross(const SrcT* a, DstT* o, float s, size_t n, int src_fmt, int dst_fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        float val = ldf<SrcT>(a, i, src_fmt) / s;
        stf<DstT>(o, i, val, dst_fmt);
    }
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

// ✅ NEW: Cross-type scalar/tensor division
template<typename SrcT, typename DstT>
__global__ void k_div_copy_scalar_tensor_cross(const SrcT* a, DstT* o, float s, size_t n, 
                                                 int src_fmt, int dst_fmt, int* flag) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        if constexpr (std::is_integral_v<SrcT>) {
            if (src_fmt == 0 && a[i] == (SrcT)0) { 
                if (flag) atomicExch(flag, 1); 
            }
        }
        float val = s / ldf<SrcT>(a, i, src_fmt);
        stf<DstT>(o, i, val, dst_fmt);
    }
}

// ============================================================================
// COMPARISON KERNELS (Write to uint8_t* - unchanged)
// ============================================================================
template<typename T>
__global__ void k_eq_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (ldf<T>(a, i, fmt) == s) ? 1 : 0;
}

template<typename T>
__global__ void k_neq_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (ldf<T>(a, i, fmt) != s) ? 1 : 0;
}

template<typename T>
__global__ void k_geq_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (ldf<T>(a, i, fmt) >= s) ? 1 : 0;
}

template<typename T>
__global__ void k_leq_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (ldf<T>(a, i, fmt) <= s) ? 1 : 0;
}

template<typename T>
__global__ void k_lt_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (ldf<T>(a, i, fmt) < s) ? 1 : 0;
}

template<typename T>
__global__ void k_gt_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (ldf<T>(a, i, fmt) > s) ? 1 : 0;
}

template<typename T>
__global__ void k_s_geq_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (s >= ldf<T>(a, i, fmt)) ? 1 : 0;
}

template<typename T>
__global__ void k_s_leq_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (s <= ldf<T>(a, i, fmt)) ? 1 : 0;
}

template<typename T>
__global__ void k_s_lt_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (s < ldf<T>(a, i, fmt)) ? 1 : 0;
}

template<typename T>
__global__ void k_s_gt_copy(const T* a, uint8_t* o, float s, size_t n, int fmt) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = (s > ldf<T>(a, i, fmt)) ? 1 : 0;
}

// ============================================================================
// LAUNCH HELPERS
// ============================================================================
template <typename T, typename Kernel>
inline void launch_copy(const Tensor& a, Tensor& out, double s, Kernel k, cudaStream_t stream) {
    const size_t n = a.numel();
    const dim3 block = dim3(256), grid = pick_grid(n, block);
    const int fmt = half_fmt(a.dtype());
    k<<<grid, block, 0, stream>>>(a.data<T>(), out.data<T>(), (float)s, n, fmt);
    ckerr("scalar copy");
}

template <typename T, typename Kernel>
inline void launch_inplace(Tensor& t, double s, Kernel k, cudaStream_t stream) {
    const size_t n = t.numel();
    const dim3 block = dim3(256), grid = pick_grid(n, block);
    const int fmt = half_fmt(t.dtype());
    k<<<grid, block, 0, stream>>>(t.data<T>(), (float)s, n, fmt);
    ckerr("scalar inplace");
}

template <typename T, typename Kernel>
inline void launch_copy_to_bool(const Tensor& a, Tensor& out, double s, Kernel k, cudaStream_t stream) {
    const size_t n = a.numel();
    const dim3 block = dim3(256), grid = pick_grid(n, block);
    const int fmt = half_fmt(a.dtype());
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    k<<<grid, block, 0, stream>>>(a.data<T>(), out_ptr, (float)s, n, fmt);
    ckerr("scalar comparison copy");
}

} // anon namespace

// ============================================================================
// PUBLIC CUDA BACKEND - ARITHMETIC (In-place)
// ============================================================================
void cuda_add_inplace(Tensor& t, double s, cudaStream_t stream) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_add_inplace<T>, stream); });
}
void cuda_sub_inplace(Tensor& t, double s, cudaStream_t stream) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_sub_inplace<T>, stream); });
}
void cuda_mul_inplace(Tensor& t, double s, cudaStream_t stream) {
    dispatch_by_dtype(t.dtype(), [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_mul_inplace<T>, stream); });
}

// ✅ FIXED: Check promotion before in-place division
void cuda_div_inplace(Tensor& t, double s, cudaStream_t stream) {
    Dtype dt = t.dtype();
    Dtype promoted_dt = get_division_output_dtype(dt);
    
    if (promoted_dt != dt) {
        throw std::runtime_error(
            "In-place division /= requires float dtype. Input is " + 
            get_dtype_name(dt) + " but needs " + get_dtype_name(promoted_dt)
        );
    }
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d); launch_inplace<T>(t, s, k_div_inplace<T>, stream); });
}

// ============================================================================
// PUBLIC CUDA BACKEND - ARITHMETIC (Copy)
// ============================================================================
Tensor cuda_add_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_add_copy<T>, stream); });
    return out;
}
Tensor cuda_sub_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_sub_copy<T>, stream); });
    return out;
}
Tensor cuda_mul_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_mul_copy<T>, stream); });
    return out;
}

// ✅ FIXED: Division promotes to Float32 for integers/bool
Tensor cuda_div_copy(const Tensor& a, double s, cudaStream_t stream) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = get_division_output_dtype(input_dt);
    
    Tensor out(a.shape(), output_dt, a.device(), a.requires_grad());
    
    if (input_dt == output_dt) {
        // Same type - use original kernel
        dispatch_by_dtype(input_dt, [&](auto d){ 
            using T = decltype(d); 
            launch_copy<T>(a, out, s, k_div_copy<T>, stream); 
        });
    } else {
        // Cross-type (Int16/Bool → Float32)
        const size_t n = a.numel();
        const dim3 block = dim3(256), grid = pick_grid(n, block);
        const int src_fmt = half_fmt(input_dt);
        const int dst_fmt = half_fmt(output_dt);
        
        dispatch_by_dtype(input_dt, [&](auto d_in) {
            using SrcT = decltype(d_in);
            dispatch_by_dtype(output_dt, [&](auto d_out) {
                using DstT = decltype(d_out);
                k_div_copy_cross<SrcT, DstT><<<grid, block, 0, stream>>>(
                    a.data<SrcT>(), out.data<DstT>(), (float)s, n, src_fmt, dst_fmt
                );
            });
        });
        ckerr("scalar div copy cross-type");
    }
    
    return out;
}

Tensor cuda_sub_copy_scalar_tensor(double s, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy<T>(a, out, s, k_sub_copy_scalar_tensor<T>, stream); });
    return out;
}

// ✅ FIXED: Scalar / Tensor also promotes
Tensor cuda_div_copy_scalar_tensor(double s, const Tensor& a, cudaStream_t stream) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = get_division_output_dtype(input_dt);
    
    Tensor out(a.shape(), output_dt, a.device(), a.requires_grad());
    
    const size_t n = a.numel();
    const dim3 block = dim3(256), grid = pick_grid(n, block);
    
    int host_flag = 0;
    int* dev_flag = nullptr;
    cudaMalloc(&dev_flag, sizeof(int));
    cudaMemsetAsync(dev_flag, 0, sizeof(int), stream);
    
    if (input_dt == output_dt) {
        dispatch_by_dtype(input_dt, [&](auto d){
            using T = decltype(d);
            const int fmt = half_fmt(input_dt);
            k_div_copy_scalar_tensor<T><<<grid, block, 0, stream>>>(
                a.data<T>(), out.data<T>(), (float)s, n, fmt, dev_flag
            );
        });
    } else {
        const int src_fmt = half_fmt(input_dt);
        const int dst_fmt = half_fmt(output_dt);
        
        dispatch_by_dtype(input_dt, [&](auto d_in) {
            using SrcT = decltype(d_in);
            dispatch_by_dtype(output_dt, [&](auto d_out) {
                using DstT = decltype(d_out);
                k_div_copy_scalar_tensor_cross<SrcT, DstT><<<grid, block, 0, stream>>>(
                    a.data<SrcT>(), out.data<DstT>(), (float)s, n, src_fmt, dst_fmt, dev_flag
                );
            });
        });
    }
    
    ckerr("k_div_copy_scalar_tensor");
    cudaMemcpyAsync(&host_flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaFree(dev_flag);
    
    if (host_flag) throw std::runtime_error("Division by zero in scalar / tensor");
    
    return out;
}

// ============================================================================
// PUBLIC CUDA BACKEND - COMPARISON OPERATORS (unchanged)
// ============================================================================
Tensor cuda_eq_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_eq_copy<T>, stream); });
    return out;
}

Tensor cuda_neq_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_neq_copy<T>, stream); });
    return out;
}

Tensor cuda_geq_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_geq_copy<T>, stream); });
    return out;
}

Tensor cuda_leq_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_leq_copy<T>, stream); });
    return out;
}

Tensor cuda_gt_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_gt_copy<T>, stream); });
    return out;
}

Tensor cuda_lt_copy(const Tensor& a, double s, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_lt_copy<T>, stream); });
    return out;
}

Tensor cuda_s_geq_copy(double s, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_s_geq_copy<T>, stream); });
    return out;
}

Tensor cuda_s_leq_copy(double s, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_s_leq_copy<T>, stream); });
    return out;
}

Tensor cuda_s_gt_copy(double s, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_s_gt_copy<T>, stream); });
    return out;
}

Tensor cuda_s_lt_copy(double s, const Tensor& a, cudaStream_t stream) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    dispatch_by_dtype(a.dtype(), [&](auto d){ using T = decltype(d); launch_copy_to_bool<T>(a, out, s, k_s_lt_copy<T>, stream); });
    return out;
}

} // namespace OwnTensor
#endif // WITH_CUDA