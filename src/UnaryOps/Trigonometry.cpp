// UnaryTrigOps_tbb_no_sleef.cpp
// oneTBB + SIMD version without SLEEF

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <cstring>
#include <immintrin.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "../include/Tensor.h"
#include "fp16_bf16_convert.h"

// --------------------------
// Unary operation kinds
// --------------------------
enum class UnaryKind {
    Sin, Cos, Tan, Asin, Acos, Atan,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
};

// --------------------------
// Scalar fallback
// --------------------------
template <typename T, UnaryKind K>
static inline T scalar_apply(T x) {
    if constexpr (K == UnaryKind::Sin)   { using std::sin;   return static_cast<T>(sin(x)); }
    if constexpr (K == UnaryKind::Cos)   { using std::cos;   return static_cast<T>(cos(x)); }
    if constexpr (K == UnaryKind::Tan)   { using std::tan;   return static_cast<T>(tan(x)); }
    if constexpr (K == UnaryKind::Asin)  { using std::asin;  return static_cast<T>(asin(x)); }
    if constexpr (K == UnaryKind::Acos)  { using std::acos;  return static_cast<T>(acos(x)); }
    if constexpr (K == UnaryKind::Atan)  { using std::atan;  return static_cast<T>(atan(x)); }
    if constexpr (K == UnaryKind::Sinh)  { using std::sinh;  return static_cast<T>(sinh(x)); }
    if constexpr (K == UnaryKind::Cosh)  { using std::cosh;  return static_cast<T>(cosh(x)); }
    if constexpr (K == UnaryKind::Tanh)  { using std::tanh;  return static_cast<T>(tanh(x)); }
    if constexpr (K == UnaryKind::Asinh) { using std::asinh; return static_cast<T>(asinh(x)); }
    if constexpr (K == UnaryKind::Acosh) { using std::acosh; return static_cast<T>(acosh(x)); }
    if constexpr (K == UnaryKind::Atanh) { using std::atanh; return static_cast<T>(atanh(x)); }
    return T(0);
}

// --------------------------
// Grain heuristic
// --------------------------
static inline size_t default_parallel_grain() {
    return 1 << 14; // 16384 elements
}

// --------------------------
// Core SIMD kernel for float/double
// --------------------------
template <typename T, UnaryKind K>
static void apply_unary_kernel(const T* in, T* out, size_t n) {
    size_t i = 0;

#if defined(__AVX512F__)
    if constexpr (std::is_same<T,float>::value) {
        const size_t L = 16;
        for (; i + L <= n; i += L) {
            __m512 v = _mm512_loadu_ps(in + i);
            float tmp[16];
            _mm512_storeu_ps(tmp, v);
            for (size_t j = 0; j < 16; ++j) tmp[j] = scalar_apply<float,K>(tmp[j]);
            _mm512_storeu_ps(out + i, _mm512_loadu_ps(tmp));
        }
    } else if constexpr (std::is_same<T,double>::value) {
        const size_t L = 8;
        for (; i + L <= n; i += L) {
            __m512d v = _mm512_loadu_pd(in + i);
            double tmp[8];
            _mm512_storeu_pd(tmp, v);
            for (size_t j = 0; j < 8; ++j) tmp[j] = scalar_apply<double,K>(tmp[j]);
            _mm512_storeu_pd(out + i, _mm512_loadu_pd(tmp));
        }
    }
#elif defined(__AVX2__)
    if constexpr (std::is_same<T,float>::value) {
        const size_t L = 8;
        for (; i + L <= n; i += L) {
            __m256 v = _mm256_loadu_ps(in + i);
            float tmp[8];
            _mm256_storeu_ps(tmp, v);
            for (size_t j = 0; j < 8; ++j) tmp[j] = scalar_apply<float,K>(tmp[j]);
            _mm256_storeu_ps(out + i, _mm256_loadu_ps(tmp));
        }
    } else if constexpr (std::is_same<T,double>::value) {
        const size_t L = 4;
        for (; i + L <= n; i += L) {
            __m256d v = _mm256_loadu_pd(in + i);
            double tmp[4];
            _mm256_storeu_pd(tmp, v);
            for (size_t j = 0; j < 4; ++j) tmp[j] = scalar_apply<double,K>(tmp[j]);
            _mm256_storeu_pd(out + i, _mm256_loadu_pd(tmp));
        }
    }
#elif defined(__SSE2__)
    if constexpr (std::is_same<T,float>::value) {
        const size_t L = 4;
        for (; i + L <= n; i += L) {
            __m128 v = _mm_loadu_ps(in + i);
            float tmp[4];
            _mm_storeu_ps(tmp, v);
            for (size_t j = 0; j < 4; ++j) tmp[j] = scalar_apply<float,K>(tmp[j]);
            _mm_storeu_ps(out + i, _mm_loadu_ps(tmp));
        }
    } else if constexpr (std::is_same<T,double>::value) {
        const size_t L = 2;
        for (; i + L <= n; i += L) {
            __m128d v = _mm_loadu_pd(in + i);
            double tmp[2];
            _mm_storeu_pd(tmp, v);
            for (size_t j = 0; j < 2; ++j) tmp[j] = scalar_apply<double,K>(tmp[j]);
            _mm_storeu_pd(out + i, _mm_loadu_pd(tmp));
        }
    }
#endif

    // 8x unrolled scalar tail
    for (; i + 7 < n; i += 8) {
        out[i+0] = scalar_apply<T,K>(in[i+0]);
        out[i+1] = scalar_apply<T,K>(in[i+1]);
        out[i+2] = scalar_apply<T,K>(in[i+2]);
        out[i+3] = scalar_apply<T,K>(in[i+3]);
        out[i+4] = scalar_apply<T,K>(in[i+4]);
        out[i+5] = scalar_apply<T,K>(in[i+5]);
        out[i+6] = scalar_apply<T,K>(in[i+6]);
        out[i+7] = scalar_apply<T,K>(in[i+7]);
    }
    for (; i < n; ++i) out[i] = scalar_apply<T,K>(in[i]);
}

// --------------------------
// f16/bf16 <-> f32 bulk converters
// --------------------------
static inline void load_f16_to_f32(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = float16_to_float(in[i]);
}
static inline void store_f32_to_f16(const float* in, uint16_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = float_to_float16(in[i]);
}
static inline void load_bf16_to_f32(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = bfloat16_to_float(in[i]);
}
static inline void store_f32_to_bf16(const float* in, uint16_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = float_to_bfloat16(in[i]);
}

// --------------------------
// Tensor adapters (TBB version)
// --------------------------
template <UnaryKind K>
static void unary_inplace_impl(Tensor& x) {
    if (!x.is_contiguous()) throw std::runtime_error("Non-contiguous tensors not supported.");
    const size_t n = x.numel();
    size_t grain = default_parallel_grain();

    if (x.dtype() == Dtype::Float32) {
        float* p = reinterpret_cast<float*>(x.data());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            apply_unary_kernel<float,K>(p + r.begin(), p + r.begin(), r.end() - r.begin());
        });
        return;
    }
    if (x.dtype() == Dtype::Float64) {
        double* p = reinterpret_cast<double*>(x.data());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            apply_unary_kernel<double,K>(p + r.begin(), p + r.begin(), r.end() - r.begin());
        });
        return;
    }

    // f16/bf16
    if (x.dtype() == Dtype::Float16 || x.dtype() == Dtype::Bfloat16) {
        uint16_t* raw = reinterpret_cast<uint16_t*>(x.data());
        std::vector<float> tmp(n);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            size_t s = r.begin(), e = r.end();
            if (x.dtype() == Dtype::Float16) load_f16_to_f32(raw + s, tmp.data() + s, e - s);
            else load_bf16_to_f32(raw + s, tmp.data() + s, e - s);
            apply_unary_kernel<float,K>(tmp.data() + s, tmp.data() + s, e - s);
            if (x.dtype() == Dtype::Float16) store_f32_to_f16(tmp.data() + s, raw + s, e - s);
            else store_f32_to_bf16(tmp.data() + s, raw + s, e - s);
        });
        return;
    }

    throw std::runtime_error("Unsupported dtype for in-place trig ops.");
}

template <UnaryKind K>
static Tensor unary_out_impl(const Tensor& x) {
    if (!x.is_contiguous()) throw std::runtime_error("Non-contiguous tensors not supported.");
    const size_t n = x.numel();
    size_t grain = default_parallel_grain();

    if (x.dtype() == Dtype::Float32) {
        Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float32, x.device(), x.requires_grad()});
        const float* in = reinterpret_cast<const float*>(x.data());
        float* out = reinterpret_cast<float*>(y.data());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            apply_unary_kernel<float,K>(in + r.begin(), out + r.begin(), r.end() - r.begin());
        });
        return y;
    }
    if (x.dtype() == Dtype::Float64) {
        Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float64, x.device(), x.requires_grad()});
        const double* in = reinterpret_cast<const double*>(x.data());
        double* out = reinterpret_cast<double*>(y.data());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            apply_unary_kernel<double,K>(in + r.begin(), out + r.begin(), r.end() - r.begin());
        });
        return y;
    }

    // f16/bf16
    if (x.dtype() == Dtype::Float16 || x.dtype() == Dtype::Bfloat16) {
        Tensor y(Shape{x.shape()}, TensorOptions{x.dtype(), x.device(), x.requires_grad()});
        const uint16_t* in_raw = reinterpret_cast<const uint16_t*>(x.data());
        uint16_t* out_raw = reinterpret_cast<uint16_t*>(y.data());
        std::vector<float> tmp_in(n), tmp_out(n);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            size_t s = r.begin(), e = r.end();
            if (x.dtype() == Dtype::Float16) load_f16_to_f32(in_raw + s, tmp_in.data() + s, e - s);
            else load_bf16_to_f32(in_raw + s, tmp_in.data() + s, e - s);
            apply_unary_kernel<float,K>(tmp_in.data() + s, tmp_out.data() + s, e - s);
            if (x.dtype() == Dtype::Float16) store_f32_to_f16(tmp_out.data() + s, out_raw + s, e - s);
            else store_f32_to_bf16(tmp_out.data() + s, out_raw + s, e - s);
        });
        return y;
    }
        // Integers → promote: int32→float32, int64→float64 ; output is float
    if (x.dtype() == Dtype::Int32) {
        Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float32, x.device(), x.requires_grad()});
        const int32_t* in = reinterpret_cast<const int32_t*>(x.data());
        float* out = reinterpret_cast<float*>(y.data());
        const size_t n = x.numel();
        size_t grain = default_parallel_grain();
        // Convert & compute in parallel in blocks to minimize memory
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            size_t s = r.begin(), e = r.end();
            // temporary buffer for block
            std::vector<float> tmp_in(e - s);
            for (size_t i = 0; i < e - s; ++i) tmp_in[i] = static_cast<float>(in[s + i]);
            apply_unary_kernel<float, K>(tmp_in.data(), tmp_in.data(), e - s);
            // copy results into output
            std::memcpy(out + s, tmp_in.data(), (e - s) * sizeof(float));
        });
        return y;
    }

    if (x.dtype() == Dtype::Int64) {
        Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float64, x.device(), x.requires_grad()});
        const int64_t* in = reinterpret_cast<const int64_t*>(x.data());
        double* out = reinterpret_cast<double*>(y.data());
        const size_t n = x.numel();
        size_t grain = default_parallel_grain();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain), [&](const tbb::blocked_range<size_t>& r){
            size_t s = r.begin(), e = r.end();
            std::vector<double> tmp_in(e - s);
            for (size_t i = 0; i < e - s; ++i) tmp_in[i] = static_cast<double>(in[s + i]);
            apply_unary_kernel<double, K>(tmp_in.data(), tmp_in.data(), e - s);
            std::memcpy(out + s, tmp_in.data(), (e - s) * sizeof(double));
        });
        return y;
    }


    throw std::runtime_error("Unsupported dtype for out-of-place trig ops.");
}

// --------------------------
// Public API
// --------------------------
Tensor sin  (const Tensor& x){ return unary_out_impl<UnaryKind::Sin>(x); }  void sin_  (Tensor& x){ unary_inplace_impl<UnaryKind::Sin>(x); }
Tensor cos  (const Tensor& x){ return unary_out_impl<UnaryKind::Cos>(x); }  void cos_  (Tensor& x){ unary_inplace_impl<UnaryKind::Cos>(x); }
Tensor tan  (const Tensor& x){ return unary_out_impl<UnaryKind::Tan>(x); }  void tan_  (Tensor& x){ unary_inplace_impl<UnaryKind::Tan>(x); }
Tensor asin (const Tensor& x){ return unary_out_impl<UnaryKind::Asin>(x);}  void asin_ (Tensor& x){ unary_inplace_impl<UnaryKind::Asin>(x);} 
Tensor acos (const Tensor& x){ return unary_out_impl<UnaryKind::Acos>(x);}  void acos_ (Tensor& x){ unary_inplace_impl<UnaryKind::Acos>(x);} 
Tensor atan (const Tensor& x){ return unary_out_impl<UnaryKind::Atan>(x);}  void atan_ (Tensor& x){ unary_inplace_impl<UnaryKind::Atan>(x);} 
Tensor sinh (const Tensor& x){ return unary_out_impl<UnaryKind::Sinh>(x);}  void sinh_ (Tensor& x){ unary_inplace_impl<UnaryKind::Sinh>(x);} 
Tensor cosh (const Tensor& x){ return unary_out_impl<UnaryKind::Cosh>(x);}  void cosh_ (Tensor& x){ unary_inplace_impl<UnaryKind::Cosh>(x);} 
Tensor tanh (const Tensor& x){ return unary_out_impl<UnaryKind::Tanh>(x);}  void tanh_ (Tensor& x){ unary_inplace_impl<UnaryKind::Tanh>(x);} 
Tensor asinh(const Tensor& x){ return unary_out_impl<UnaryKind::Asinh>(x);} void asinh_(Tensor& x){ unary_inplace_impl<UnaryKind::Asinh>(x);} 
Tensor acosh(const Tensor& x){ return unary_out_impl<UnaryKind::Acosh>(x);} void acosh_(Tensor& x){ unary_inplace_impl<UnaryKind::Acosh>(x);} 
Tensor atanh(const Tensor& x){ return unary_out_impl<UnaryKind::Atanh>(x);} void atanh_(Tensor& x){ unary_inplace_impl<UnaryKind::Atanh>(x);} 
