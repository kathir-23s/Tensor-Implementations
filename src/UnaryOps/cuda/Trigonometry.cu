// src/UnaryOps/Trigonometry.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if __CUDACC_VER_MAJOR__ >= 11
  #include <cuda_bf16.h>
#endif

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"

namespace OwnTensor {

// ===================================================================
// Trig selector
// ===================================================================
enum class Trig {
    Sin, Cos, Tan, Asin, Acos, Atan,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
};

static inline bool is_int_dtype(Dtype dt) {
    return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
}

// helper that "returns" a T but always throws (fixes return type deduction)
template <typename T>
static inline T raise_unsupported(const char* msg) {
    throw std::runtime_error(msg);
}

// ===================================================================
// Device math
// ===================================================================
template <Trig K> __device__ inline float  dev_apply(float  x) {
    if constexpr (K==Trig::Sin  ) return sinf (x);
    if constexpr (K==Trig::Cos  ) return cosf (x);
    if constexpr (K==Trig::Tan  ) return tanf (x);
    if constexpr (K==Trig::Asin ) return asinf(x);
    if constexpr (K==Trig::Acos ) return acosf(x);
    if constexpr (K==Trig::Atan ) return atanf(x);
    if constexpr (K==Trig::Sinh ) return sinhf(x);
    if constexpr (K==Trig::Cosh ) return coshf(x);
    if constexpr (K==Trig::Tanh ) return tanhf(x);
    if constexpr (K==Trig::Asinh) return asinhf(x);
    if constexpr (K==Trig::Acosh) return acoshf(x);
    if constexpr (K==Trig::Atanh) return atanhf(x);
    return 0.0f;
}
template <Trig K> __device__ inline double dev_apply(double x) {
    if constexpr (K==Trig::Sin  ) return sin  (x);
    if constexpr (K==Trig::Cos  ) return cos  (x);
    if constexpr (K==Trig::Tan  ) return tan  (x);
    if constexpr (K==Trig::Asin ) return asin (x);
    if constexpr (K==Trig::Acos ) return acos (x);
    if constexpr (K==Trig::Atan ) return atan (x);
    if constexpr (K==Trig::Sinh ) return sinh (x);
    if constexpr (K==Trig::Cosh ) return cosh (x);
    if constexpr (K==Trig::Tanh ) return tanh (x);
    if constexpr (K==Trig::Asinh) return asinh(x);
    if constexpr (K==Trig::Acosh) return acosh(x);
    if constexpr (K==Trig::Atanh) return atanh(x);
    return 0.0;
}

// ===================================================================
// Kernels
// ===================================================================
template <typename T, Trig K>
__global__ void unary_kernel_fp(const T* __restrict__ in, T* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_apply<K>(in[i]);
}

template <Trig K>
__global__ void unary_kernel_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __half2float(in[i]);
        float y = dev_apply<K>(x);
        out[i]  = __float2half(y);
    }
}

#if __CUDACC_VER_MAJOR__ >= 11
template <Trig K>
__global__ void unary_kernel_bf16(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(in[i]);
        float y = dev_apply<K>(x);
        out[i]  = __float2bfloat16(y);
    }
}
#endif

template <Trig K>
__global__ void unary_kernel_i16_to_f32(const int16_t* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_apply<K>(static_cast<float>(in[i]));
}
template <Trig K>
__global__ void unary_kernel_i32_to_f32(const int32_t* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_apply<K>(static_cast<float>(in[i]));
}
template <Trig K>
__global__ void unary_kernel_i64_to_f64(const int64_t* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_apply<K>(static_cast<double>(in[i]));
}

// ===================================================================
// Launch helpers
// ===================================================================
static inline dim3 default_block() { return dim3(256); }
static inline dim3 default_grid(size_t n) { return dim3(static_cast<unsigned>((n + 255) / 256)); }

static inline void*       data_mut(Tensor& t){ return t.data(); }
static inline const void* data_const(const Tensor& t){ return t.data(); }
static inline Tensor      make_like(const Tensor& x){ return Tensor(Shape{x.shape()}, x.dtype(), x.device(), x.requires_grad()); }

// ===================================================================
// In-place via dispatch_by_dtype
// ===================================================================
template <Trig K>
static inline void cuda_unary_inplace(Tensor& x) {
    if (!x.is_contiguous()) throw std::runtime_error("Non-contiguous tensors not supported yet.");
    if (is_int_dtype(x.dtype()))
        throw std::runtime_error("In-place trig ops not supported for integer tensors on GPU. Use out-of-place.");

    const size_t n = x.numel();
    if (n == 0) return;

    dim3 block = default_block();
    dim3 grid  = default_grid(n);

    dispatch_by_dtype(x.dtype(), [&](auto tag){
        using Tag = std::decay_t<decltype(tag)>;
        cudaError_t st;

        if constexpr (std::is_same_v<Tag, float>) {
            auto* p = reinterpret_cast<float*>(data_mut(x));
            unary_kernel_fp<float,K><<<grid, block>>>(p, p, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        }
        else if constexpr (std::is_same_v<Tag, double>) {
            auto* p = reinterpret_cast<double*>(data_mut(x));
            unary_kernel_fp<double,K><<<grid, block>>>(p, p, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        }
        else if constexpr (std::is_same_v<Tag, __half>) {
            auto* p = reinterpret_cast<__half*>(data_mut(x));
            unary_kernel_f16<K><<<grid, block>>>(p, p, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        }
#if __CUDACC_VER_MAJOR__ >= 11
        else if constexpr (std::is_same_v<Tag, __nv_bfloat16>) {
            auto* p = reinterpret_cast<__nv_bfloat16*>(data_mut(x));
            unary_kernel_bf16<K><<<grid, block>>>(p, p, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
        }
#endif
        else if constexpr (std::is_same_v<Tag, int16_t> ||
                           std::is_same_v<Tag, int32_t> ||
                           std::is_same_v<Tag, int64_t>) {
            (void)grid; (void)block;
            throw std::runtime_error("In-place trig ops not supported for integer tensors on GPU. Use out-of-place.");
        }
        else {
            (void)grid; (void)block;
            throw std::runtime_error("Unsupported dtype in cuda_unary_inplace");
        }
    });
}

// ===================================================================
// Out-of-place via dispatch_by_dtype
// ===================================================================
template <Trig K>
static inline Tensor cuda_unary_out(const Tensor& x) {
    if (!x.is_contiguous()) throw std::runtime_error("Non-contiguous tensors not supported yet.");
    const size_t n = x.numel();
    dim3 block = default_block();
    dim3 grid  = default_grid(n);

    return dispatch_by_dtype(x.dtype(), [&](auto tag){
        using Tag = std::decay_t<decltype(tag)>;
        cudaError_t st;

        if constexpr (std::is_same_v<Tag, float>) {
            Tensor y = make_like(x);
            auto* in  = reinterpret_cast<const float*>(data_const(x));
            auto* out = reinterpret_cast<float*>(data_mut(y));
            unary_kernel_fp<float,K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
        else if constexpr (std::is_same_v<Tag, double>) {
            Tensor y = make_like(x);
            auto* in  = reinterpret_cast<const double*>(data_const(x));
            auto* out = reinterpret_cast<double*>(data_mut(y));
            unary_kernel_fp<double,K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
        else if constexpr (std::is_same_v<Tag, __half>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float16, x.device(), x.requires_grad()});
            auto* in  = reinterpret_cast<const __half*>(data_const(x));
            auto* out = reinterpret_cast<__half*>(data_mut(y));
            unary_kernel_f16<K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
#if __CUDACC_VER_MAJOR__ >= 11
        else if constexpr (std::is_same_v<Tag, __nv_bfloat16>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Bfloat16, x.device(), x.requires_grad()});
            auto* in  = reinterpret_cast<const __nv_bfloat16*>(data_const(x));
            auto* out = reinterpret_cast<__nv_bfloat16*>(data_mut(y));
            unary_kernel_bf16<K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
#endif
        else if constexpr (std::is_same_v<Tag, int16_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float32, x.device(), x.requires_grad()});
            auto* in  = reinterpret_cast<const int16_t*>(data_const(x));
            auto* out = reinterpret_cast<float*>(data_mut(y));
            unary_kernel_i16_to_f32<K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
        else if constexpr (std::is_same_v<Tag, int32_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float32, x.device(), x.requires_grad()});
            auto* in  = reinterpret_cast<const int32_t*>(data_const(x));
            auto* out = reinterpret_cast<float*>(data_mut(y));
            unary_kernel_i32_to_f32<K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
        else if constexpr (std::is_same_v<Tag, int64_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float64, x.device(), x.requires_grad()});
            auto* in  = reinterpret_cast<const int64_t*>(data_const(x));
            auto* out = reinterpret_cast<double*>(data_mut(y));
            unary_kernel_i64_to_f64<K><<<grid, block>>>(in, out, n);
            st = cudaGetLastError(); if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
            return y;
        }
        else {
            return raise_unsupported<Tensor>("Unsupported dtype in cuda_unary_out");
        }
    });
}

// ===================================================================
// Public API (GPU) — mirrors CPU names
// ===================================================================
Tensor sin_cuda  (const Tensor& x){ return cuda_unary_out<Trig::Sin>(x); }   void sin__cuda  (Tensor& x){ cuda_unary_inplace<Trig::Sin>(x); }
Tensor cos_cuda  (const Tensor& x){ return cuda_unary_out<Trig::Cos>(x); }   void cos__cuda  (Tensor& x){ cuda_unary_inplace<Trig::Cos>(x); }
Tensor tan_cuda  (const Tensor& x){ return cuda_unary_out<Trig::Tan>(x); }   void tan__cuda  (Tensor& x){ cuda_unary_inplace<Trig::Tan>(x); }
Tensor asin_cuda (const Tensor& x){ return cuda_unary_out<Trig::Asin>(x);}   void asin__cuda (Tensor& x){ cuda_unary_inplace<Trig::Asin>(x);} 
Tensor acos_cuda (const Tensor& x){ return cuda_unary_out<Trig::Acos>(x);}   void acos__cuda (Tensor& x){ cuda_unary_inplace<Trig::Acos>(x);} 
Tensor atan_cuda (const Tensor& x){ return cuda_unary_out<Trig::Atan>(x);}   void atan__cuda (Tensor& x){ cuda_unary_inplace<Trig::Atan>(x);} 
Tensor sinh_cuda (const Tensor& x){ return cuda_unary_out<Trig::Sinh>(x);}   void sinh__cuda (Tensor& x){ cuda_unary_inplace<Trig::Sinh>(x);} 
Tensor cosh_cuda (const Tensor& x){ return cuda_unary_out<Trig::Cosh>(x);}   void cosh__cuda (Tensor& x){ cuda_unary_inplace<Trig::Cosh>(x);} 
Tensor tanh_cuda (const Tensor& x){ return cuda_unary_out<Trig::Tanh>(x);}   void tanh__cuda (Tensor& x){ cuda_unary_inplace<Trig::Tanh>(x);} 
Tensor asinh_cuda(const Tensor& x){ return cuda_unary_out<Trig::Asinh>(x);}  void asinh__cuda(Tensor& x){ cuda_unary_inplace<Trig::Asinh>(x);} 
Tensor acosh_cuda(const Tensor& x){ return cuda_unary_out<Trig::Acosh>(x);}  void acosh__cuda(Tensor& x){ cuda_unary_inplace<Trig::Acosh>(x);} 
Tensor atanh_cuda(const Tensor& x){ return cuda_unary_out<Trig::Atanh>(x);}  void atanh__cuda(Tensor& x){ cuda_unary_inplace<Trig::Atanh>(x);} 

} // namespace OwnTensor
