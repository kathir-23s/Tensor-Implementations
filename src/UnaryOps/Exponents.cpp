// ExpLogKernels_sleef.cpp
// Out-of-place kernels for exp/log family (float/double) using SLEEF SIMD + OpenMP.
// In-place is achieved by dispatcher by passing a==b.
//
// Build:
//   g++ -O3 -std=c++20 -fopenmp -I/path/to/sleef/include \
//       -c src/UnaryOps/ExpLogKernels_sleef.cpp && \
//   ...link with -L/path/to/sleef/lib -lsleef -lm
//
// Notes:
//   - Default: SLEEF u10 accuracy (fast). Define EXLOG_SLEEF_U35 for higher accuracy.
//   - Uses AVX-512 / AVX2 / SSE2 paths when available, otherwise falls back to libm.
//   - Threading: OpenMP over chunks (threshold 4096).

#include <cstddef>
#include <type_traits>
#include <immintrin.h>
#include <sleef.h>
#include <cmath>

#if defined(_OPENMP)
  #include <omp.h>
  #define PARALLEL_FOR(n) _Pragma("omp parallel for if(n > 4096) schedule(static)")
#else
  #define PARALLEL_FOR(n)
#endif

namespace OwnTensor {
namespace exp_log_detail {

// =========================================
// Select SLEEF accuracy family
// =========================================
#if defined(EXLOG_SLEEF_U35)
  // AVX-512
  #define SLEEF_EXPF16(v)    Sleef_expf16_u35avx512f(v)
  #define SLEEF_EXP2F16(v)   Sleef_exp2f16_u35avx512f(v)
  #define SLEEF_LOGF16(v)    Sleef_logf16_u35avx512f(v)
  #define SLEEF_LOG2F16(v)   Sleef_log2f16_u35avx512f(v)
  #define SLEEF_LOG10F16(v)  Sleef_log10f16_u35avx512f(v)
  #define SLEEF_EXPD8(v)     Sleef_expd8_u35avx512f(v)
  #define SLEEF_EXP2D8(v)    Sleef_exp2d8_u35avx512f(v)
  #define SLEEF_LOGD8(v)     Sleef_logd8_u35avx512f(v)
  #define SLEEF_LOG2D8(v)    Sleef_log2d8_u35avx512f(v)
  #define SLEEF_LOG10D8(v)   Sleef_log10d8_u35avx512f(v)
  // AVX2
  #define SLEEF_EXPF8(v)     Sleef_expf8_u35avx2(v)
  #define SLEEF_EXP2F8(v)    Sleef_exp2f8_u35avx2(v)
  #define SLEEF_LOGF8(v)     Sleef_logf8_u35avx2(v)
  #define SLEEF_LOG2F8(v)    Sleef_log2f8_u35avx2(v)
  #define SLEEF_LOG10F8(v)   Sleef_log10f8_u35avx2(v)
  #define SLEEF_EXPD4(v)     Sleef_expd4_u35avx2(v)
  #define SLEEF_EXP2D4(v)    Sleef_exp2d4_u35avx2(v)
  #define SLEEF_LOGD4(v)     Sleef_logd4_u35avx2(v)
  #define SLEEF_LOG2D4(v)    Sleef_log2d4_u35avx2(v)
  #define SLEEF_LOG10D4(v)   Sleef_log10d4_u35avx2(v)
  // SSE2
  #define SLEEF_EXPF4(v)     Sleef_expf4_u35sse2(v)
  #define SLEEF_EXP2F4(v)    Sleef_exp2f4_u35sse2(v)
  #define SLEEF_LOGF4(v)     Sleef_logf4_u35sse2(v)
  #define SLEEF_LOG2F4(v)    Sleef_log2f4_u35sse2(v)
  #define SLEEF_LOG10F4(v)   Sleef_log10f4_u35sse2(v)
  #define SLEEF_EXPD2(v)     Sleef_expd2_u35sse2(v)
  #define SLEEF_EXP2D2(v)    Sleef_exp2d2_u35sse2(v)
  #define SLEEF_LOGD2(v)     Sleef_logd2_u35sse2(v)
  #define SLEEF_LOG2D2(v)    Sleef_log2d2_u35sse2(v)
  #define SLEEF_LOG10D2(v)   Sleef_log10d2_u35sse2(v)
#else
  // u10 (default)
  // AVX-512
  #define SLEEF_EXPF16(v)    Sleef_expf16_u10avx512f(v)
  #define SLEEF_EXP2F16(v)   Sleef_exp2f16_u10avx512f(v)
  #define SLEEF_LOGF16(v)    Sleef_logf16_u10avx512f(v)
  #define SLEEF_LOG2F16(v)   Sleef_log2f16_u10avx512f(v)
  #define SLEEF_LOG10F16(v)  Sleef_log10f16_u10avx512f(v)
  #define SLEEF_EXPD8(v)     Sleef_expd8_u10avx512f(v)
  #define SLEEF_EXP2D8(v)    Sleef_exp2d8_u10avx512f(v)
  #define SLEEF_LOGD8(v)     Sleef_logd8_u10avx512f(v)
  #define SLEEF_LOG2D8(v)    Sleef_log2d8_u10avx512f(v)
  #define SLEEF_LOG10D8(v)   Sleef_log10d8_u10avx512f(v)
  // AVX2
  #define SLEEF_EXPF8(v)     Sleef_expf8_u10avx2(v)
  #define SLEEF_EXP2F8(v)    Sleef_exp2f8_u10avx2(v)
  #define SLEEF_LOGF8(v)     Sleef_logf8_u10avx2(v)
  #define SLEEF_LOG2F8(v)    Sleef_log2f8_u10avx2(v)
  #define SLEEF_LOG10F8(v)   Sleef_log10f8_u10avx2(v)
  #define SLEEF_EXPD4(v)     Sleef_expd4_u10avx2(v)
  #define SLEEF_EXP2D4(v)    Sleef_exp2d4_u10avx2(v)
  #define SLEEF_LOGD4(v)     Sleef_logd4_u10avx2(v)
  #define SLEEF_LOG2D4(v)    Sleef_log2d4_u10avx2(v)
  #define SLEEF_LOG10D4(v)   Sleef_log10d4_u10avx2(v)
  // SSE2
  #define SLEEF_EXPF4(v)     Sleef_expf4_u10sse2(v)
  #define SLEEF_EXP2F4(v)    Sleef_exp2f4_u10sse2(v)
  #define SLEEF_LOGF4(v)     Sleef_logf4_u10sse2(v)
  #define SLEEF_LOG2F4(v)    Sleef_log2f4_u10sse2(v)
  #define SLEEF_LOG10F4(v)   Sleef_log10f4_u10sse2(v)
  #define SLEEF_EXPD2(v)     Sleef_expd2_u10sse2(v)
  #define SLEEF_EXP2D2(v)    Sleef_exp2d2_u10sse2(v)
  #define SLEEF_LOGD2(v)     Sleef_logd2_u10sse2(v)
  #define SLEEF_LOG2D2(v)    Sleef_log2d2_u10sse2(v)
  #define SLEEF_LOG10D2(v)   Sleef_log10d2_u10sse2(v)
#endif

// =========================================
// Op tags
// =========================================
enum class Op { Exp, Exp2, Log, Log2, Log10 };

// =========================================
// Scalar fallback (libm)
// =========================================
template <typename T, Op O> static inline T scalar_eval(T x) {
  if constexpr (O==Op::Exp)   { using std::exp;   return T(exp(x)); }
  if constexpr (O==Op::Exp2)  { using std::exp2;  return T(exp2(x)); }
  if constexpr (O==Op::Log)   { using std::log;   return T(log(x)); }
  if constexpr (O==Op::Log2)  { using std::log2;  return T(log2(x)); }
  if constexpr (O==Op::Log10) { using std::log10; return T(log10(x)); }
  return T(0);
}

// =========================================
/* SIMD vector evaluators */
// =========================================
#if defined(__AVX512F__)
template <Op O> static inline __m512  eval512(__m512  v){
  if constexpr (O==Op::Exp)   return SLEEF_EXPF16(v);
  if constexpr (O==Op::Exp2)  return SLEEF_EXP2F16(v);
  if constexpr (O==Op::Log)   return SLEEF_LOGF16(v);
  if constexpr (O==Op::Log2)  return SLEEF_LOG2F16(v);
  if constexpr (O==Op::Log10) return SLEEF_LOG10F16(v);
  return v;
}
template <Op O> static inline __m512d eval512(__m512d v){
  if constexpr (O==Op::Exp)   return SLEEF_EXPD8(v);
  if constexpr (O==Op::Exp2)  return SLEEF_EXP2D8(v);
  if constexpr (O==Op::Log)   return SLEEF_LOGD8(v);
  if constexpr (O==Op::Log2)  return SLEEF_LOG2D8(v);
  if constexpr (O==Op::Log10) return SLEEF_LOG10D8(v);
  return v;
}
#endif

#if defined(__AVX2__)
template <Op O> static inline __m256  eval256(__m256  v){
  if constexpr (O==Op::Exp)   return SLEEF_EXPF8(v);
  if constexpr (O==Op::Exp2)  return SLEEF_EXP2F8(v);
  if constexpr (O==Op::Log)   return SLEEF_LOGF8(v);
  if constexpr (O==Op::Log2)  return SLEEF_LOG2F8(v);
  if constexpr (O==Op::Log10) return SLEEF_LOG10F8(v);
  return v;
}
template <Op O> static inline __m256d eval256(__m256d v){
  if constexpr (O==Op::Exp)   return SLEEF_EXPD4(v);
  if constexpr (O==Op::Exp2)  return SLEEF_EXP2D4(v);
  if constexpr (O==Op::Log)   return SLEEF_LOGD4(v);
  if constexpr (O==Op::Log2)  return SLEEF_LOG2D4(v);
  if constexpr (O==Op::Log10) return SLEEF_LOG10D4(v);
  return v;
}
#endif

#if defined(__SSE2__)
template <Op O> static inline __m128  eval128(__m128  v){
  if constexpr (O==Op::Exp)   return SLEEF_EXPF4(v);
  if constexpr (O==Op::Exp2)  return SLEEF_EXP2F4(v);
  if constexpr (O==Op::Log)   return SLEEF_LOGF4(v);
  if constexpr (O==Op::Log2)  return SLEEF_LOG2F4(v);
  if constexpr (O==Op::Log10) return SLEEF_LOG10F4(v);
  return v;
}
template <Op O> static inline __m128d eval128(__m128d v){
  if constexpr (O==Op::Exp)   return SLEEF_EXPD2(v);
  if constexpr (O==Op::Exp2)  return SLEEF_EXP2D2(v);
  if constexpr (O==Op::Log)   return SLEEF_LOGD2(v);
  if constexpr (O==Op::Log2)  return SLEEF_LOG2D2(v);
  if constexpr (O==Op::Log10) return SLEEF_LOG10D2(v);
  return v;
}
#endif

// =========================================
// Core block (SIMD + scalar tail)
// =========================================
template <typename T, Op O>
static inline void apply_block(const T* in, T* out, size_t n){
  size_t i = 0;

#if defined(__AVX512F__)
  if constexpr (std::is_same_v<T,float>) {
    constexpr size_t L = 16;
    for (; i + L <= n; i += L) {
      __m512 v = _mm512_loadu_ps(in + i);
      v = eval512<O>(v);
      _mm512_storeu_ps(out + i, v);
    }
  } else if constexpr (std::is_same_v<T,double>) {
    constexpr size_t L = 8;
    for (; i + L <= n; i += L) {
      __m512d v = _mm512_loadu_pd(in + i);
      v = eval512<O>(v);
      _mm512_storeu_pd(out + i, v);
    }
  }
#elif defined(__AVX2__)
  if constexpr (std::is_same_v<T,float>) {
    constexpr size_t L = 8;
    for (; i + L <= n; i += L) {
      __m256 v = _mm256_loadu_ps(in + i);
      v = eval256<O>(v);
      _mm256_storeu_ps(out + i, v);
    }
  } else if constexpr (std::is_same_v<T,double>) {
    constexpr size_t L = 4;
    for (; i + L <= n; i += L) {
      __m256d v = _mm256_loadu_pd(in + i);
      v = eval256<O>(v);
      _mm256_storeu_pd(out + i, v);
    }
  }
#elif defined(__SSE2__)
  if constexpr (std::is_same_v<T,float>) {
    constexpr size_t L = 4;
    for (; i + L <= n; i += L) {
      __m128 v = _mm_loadu_ps(in + i);
      v = eval128<O>(v);
      _mm_storeu_ps(out + i, v);
    }
  } else if constexpr (std::is_same_v<T,double>) {
    constexpr size_t L = 2;
    for (; i + L <= n; i += L) {
      __m128d v = _mm_loadu_pd(in + i);
      v = eval128<O>(v);
      _mm_storeu_pd(out + i, v);
    }
  }
#endif

  // Scalar tail (and full fallback if no SIMD)
  for (; i + 7 < n; i += 8) {
    out[i+0] = scalar_eval<T,O>(in[i+0]);
    out[i+1] = scalar_eval<T,O>(in[i+1]);
    out[i+2] = scalar_eval<T,O>(in[i+2]);
    out[i+3] = scalar_eval<T,O>(in[i+3]);
    out[i+4] = scalar_eval<T,O>(in[i+4]);
    out[i+5] = scalar_eval<T,O>(in[i+5]);
    out[i+6] = scalar_eval<T,O>(in[i+6]);
    out[i+7] = scalar_eval<T,O>(in[i+7]);
  }
  for (; i < n; ++i) out[i] = scalar_eval<T,O>(in[i]);
}

// =========================================
// Public C-style kernels (void*, void*, n)
// =========================================
template <typename T, Op O>
static inline void unary_out_vec(const void* a, void* b, size_t n){
  const T* in  = static_cast<const T*>(a);
  T*       out = static_cast<T*>(b);

 const ptrdiff_t grain = 16384;
#pragma omp parallel for if(n > 4096) schedule(static)
for (ptrdiff_t s = 0; s < static_cast<ptrdiff_t>(n); s += grain) {
  size_t start = static_cast<size_t>(s);
  size_t end   = std::min(start + static_cast<size_t>(grain), n);
  apply_block<T,O>(in + start, out + start, end - start);
}
}

// ---- exported symbols used by the dispatcher ----
void exp_float_kernel   (const void* a, void* b, size_t n) { unary_out_vec<float , Op::Exp  >(a,b,n); }
void exp_double_kernel  (const void* a, void* b, size_t n) { unary_out_vec<double, Op::Exp  >(a,b,n); }
void exp2_float_kernel  (const void* a, void* b, size_t n) { unary_out_vec<float , Op::Exp2 >(a,b,n); }
void exp2_double_kernel (const void* a, void* b, size_t n) { unary_out_vec<double, Op::Exp2 >(a,b,n); }
void log_float_kernel   (const void* a, void* b, size_t n) { unary_out_vec<float , Op::Log  >(a,b,n); }
void log_double_kernel  (const void* a, void* b, size_t n) { unary_out_vec<double, Op::Log  >(a,b,n); }
void log2_float_kernel  (const void* a, void* b, size_t n) { unary_out_vec<float , Op::Log2 >(a,b,n); }
void log2_double_kernel (const void* a, void* b, size_t n) { unary_out_vec<double, Op::Log2 >(a,b,n); }
void log10_float_kernel (const void* a, void* b, size_t n) { unary_out_vec<float , Op::Log10>(a,b,n); }
void log10_double_kernel(const void* a, void* b, size_t n) { unary_out_vec<double, Op::Log10>(a,b,n); }

} // namespace exp_log_detail
}