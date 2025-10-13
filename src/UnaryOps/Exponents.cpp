// ExpLogKernels_sleef_compatible.cpp
// Out-of-place kernels for exp/log family (float/double).
// Compiles WITHOUT SLEEF by default; uses SLEEF+SIMD only with -DUSE_SLEEF=1 and sleef.h present.
//
// Build (no SLEEF):
//   g++ -O3 -std=c++20 -fopenmp -c ExpLogKernels_sleef_compatible.cpp && \
//   g++ -o your_app ... ExpLogKernels_sleef_compatible.o -lm
//
// Build (with SLEEF):
//   g++ -O3 -std=c++20 -fopenmp -DUSE_SLEEF=1 -march=native \
//       -I/path/to/sleef/include -c ExpLogKernels_sleef_compatible.cpp && \
//   g++ -o your_app ... ExpLogKernels_sleef_compatible.o -L/path/to/sleef/lib -lsleef -lm
//
// Notes:
//   - Default accuracy is SLEEF u10; define EXLOG_SLEEF_U35 for higher accuracy when SLEEF is enabled.
//   - OpenMP parallelizes over 16k chunks when n > 4096.

#include <cstddef>
#include <type_traits>
#include <cmath>
#include <algorithm> // std::min

#if defined(_OPENMP)
  #include <omp.h>
#endif

// ----------------- Strict opt-in for SLEEF (no auto-detect) -----------------
#ifndef USE_SLEEF
  #define USE_SLEEF 0
#endif

#if USE_SLEEF && defined(__has_include)
  #if __has_include(<sleef.h>)
    #define OT_HAVE_SLEEF 1
    #include <sleef.h>
    // Only include intrinsics when we might emit SIMD code:
    #if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__)
      #include <immintrin.h>
    #endif
  #else
    #define OT_HAVE_SLEEF 0
  #endif
#else
  #define OT_HAVE_SLEEF 0
#endif

namespace OwnTensor {
namespace exp_log_detail {

// =========================================
// Op tags
// =========================================
enum class Op { Exp, Exp2, Log, Log2, Log10 };

// =========================================
// Scalar fallback (libm) -- always available
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
// Optional SLEEF accuracy family & SIMD evaluators
// (compiled only when OT_HAVE_SLEEF==1)
// =========================================
#if OT_HAVE_SLEEF

// --------- Accuracy family selection (u10 default, u35 when EXLOG_SLEEF_U35) ----------
#if defined(EXLOG_SLEEF_U35)
// AVX-512 (float16x, double8x)
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
// u10 default
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

// ---------------- SIMD evaluators (guarded by ISA macros) -------------------
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
#endif // OT_HAVE_SLEEF

// =========================================
// Core block (SIMD when enabled, scalar tail always)
// =========================================
template <typename T, Op O>
static inline void apply_block(const T* in, T* out, size_t n){
  size_t i = 0;

#if OT_HAVE_SLEEF
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
#endif // OT_HAVE_SLEEF

  // Scalar tail (and full fallback if SLEEF/ISA not available)
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

  const std::ptrdiff_t grain = 16384;

#if defined(_OPENMP)
  #pragma omp parallel for if(n > 4096) schedule(static)
  for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(n); s += grain) {
    size_t start = static_cast<size_t>(s);
    size_t end   = std::min(start + static_cast<size_t>(grain), n);
    apply_block<T,O>(in + start, out + start, end - start);
  }
#else
  for (size_t start = 0; start < n; start += static_cast<size_t>(grain)) {
    size_t end = std::min(start + static_cast<size_t>(grain), n);
    apply_block<T,O>(in + start, out + start, end - start);
  }
#endif
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
} // namespace OwnTensor