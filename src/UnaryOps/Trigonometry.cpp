// Trigonometry_sleef_compatible.cpp
// SIMD (optionally SLEEF) + OpenMP kernels for trig / inverse-trig / hyperbolic / inverse-hyperbolic.
// Public API (unchanged):
//   namespace OwnTensor::trig_detail {
//     enum class UnaryKind { Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh };
//     template<typename T, UnaryKind K>
//     void apply_unary_kernel(const T* in, T* out, size_t n);
//   }
//
// Build examples:
//   (WITH SLEEF)
//   g++ -O3 -std=c++20 -fopenmp -DHAVE_SLEEF -march=native \
//       -I/path/to/sleef/include -c Trigonometry_sleef_compatible.cpp && \
//   g++ -o your_app ... Trigonometry_sleef_compatible.o -L/path/to/sleef/lib -lsleef -lm
//
//   (WITHOUT SLEEF)
//   g++ -O3 -std=c++20 -fopenmp -c Trigonometry_sleef_compatible.cpp && \
//   g++ -o your_app ... Trigonometry_sleef_compatible.o -lm
//
// Notes:
//   - Define TRIG_SLEEF_U35 for higher-accuracy SLEEF family when SLEEF is enabled.
//   - OMP_NUM_THREADS controls threading. Tune TRIG_PARALLEL_GRAIN as needed.

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <stdexcept>
#include <cmath>

#if defined(_OPENMP)
  #include <omp.h>
#endif

// ----------------------------- Feature detection -----------------------------
#ifndef HAVE_SLEEF
  // If the user didn't define HAVE_SLEEF, try detecting sleef.h presence (C++17).
  #if defined(__has_include)
    #if __has_include(<sleef.h>)
      #define HAVE_SLEEF 1
    #else
      #define HAVE_SLEEF 0
    #endif
  #else
    #define HAVE_SLEEF 0
  #endif
#endif

#if HAVE_SLEEF
  #include <sleef.h>
  // Only pull in intrinsics when we’ll actually emit SIMD calls.
  #if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__)
    #include <immintrin.h>
  #endif
#endif

namespace OwnTensor {
namespace trig_detail {

// ----------------------------------------------------------------------------------
// Public enum expected by the dispatcher
// ----------------------------------------------------------------------------------
enum class UnaryKind {
  Sin, Cos, Tan, Asin, Acos, Atan,
  Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
};

// ----------------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------------
#ifndef TRIG_PARALLEL_GRAIN
  #define TRIG_PARALLEL_GRAIN (1u << 14)  // 16384 elements per chunk
#endif

// ----------------------------------------------------------------------------------
// Scalar fallback (libm) -- always available
// ----------------------------------------------------------------------------------
template <typename T, UnaryKind K>
static inline T scalar_apply(T x) {
  if constexpr (K==UnaryKind::Sin)   { using std::sin;   return T(sin(x)); }
  if constexpr (K==UnaryKind::Cos)   { using std::cos;   return T(cos(x)); }
  if constexpr (K==UnaryKind::Tan)   { using std::tan;   return T(tan(x)); }
  if constexpr (K==UnaryKind::Asin)  { using std::asin;  return T(asin(x)); }
  if constexpr (K==UnaryKind::Acos)  { using std::acos;  return T(acos(x)); }
  if constexpr (K==UnaryKind::Atan)  { using std::atan;  return T(atan(x)); }
  if constexpr (K==UnaryKind::Sinh)  { using std::sinh;  return T(sinh(x)); }
  if constexpr (K==UnaryKind::Cosh)  { using std::cosh;  return T(cosh(x)); }
  if constexpr (K==UnaryKind::Tanh)  { using std::tanh;  return T(tanh(x)); }
  if constexpr (K==UnaryKind::Asinh) { using std::asinh; return T(asinh(x)); }
  if constexpr (K==UnaryKind::Acosh) { using std::acosh; return T(acosh(x)); }
  if constexpr (K==UnaryKind::Atanh) { using std::atanh; return T(atanh(x)); }
  return T(0);
}

// ----------------------------------------------------------------------------------
// Optional SLEEF + SIMD helpers (compiled only when HAVE_SLEEF==1)
// ----------------------------------------------------------------------------------
#if HAVE_SLEEF

// Select SLEEF accuracy family
#if defined(TRIG_SLEEF_U35)
  #define SLEEF_SINF16(v)    Sleef_sinf16_u35avx512f(v)
  #define SLEEF_COSF16(v)    Sleef_cosf16_u35avx512f(v)
  #define SLEEF_TANF16(v)    Sleef_tanf16_u35avx512f(v)
  #define SLEEF_ASINF16(v)   Sleef_asinf16_u35avx512f(v)
  #define SLEEF_ACOSF16(v)   Sleef_acosf16_u35avx512f(v)
  #define SLEEF_ATANF16(v)   Sleef_atanf16_u35avx512f(v)
  #define SLEEF_SINHF16(v)   Sleef_sinhf16_u35avx512f(v)
  #define SLEEF_COSHF16(v)   Sleef_coshf16_u35avx512f(v)
  #define SLEEF_TANHF16(v)   Sleef_tanhf16_u35avx512f(v)
  #define SLEEF_ASINHF16(v)  Sleef_asinhf16_u35avx512f(v)
  #define SLEEF_ACOSHF16(v)  Sleef_acoshf16_u35avx512f(v)
  #define SLEEF_ATANHF16(v)  Sleef_atanhf16_u35avx512f(v)

  #define SLEEF_SIND8(v)     Sleef_sind8_u35avx512f(v)
  #define SLEEF_COSD8(v)     Sleef_cosd8_u35avx512f(v)
  #define SLEEF_TAND8(v)     Sleef_tand8_u35avx512f(v)
  #define SLEEF_ASIND8(v)    Sleef_asind8_u35avx512f(v)
  #define SLEEF_ACOSD8(v)    Sleef_acosd8_u35avx512f(v)
  #define SLEEF_ATAND8(v)    Sleef_atand8_u35avx512f(v)
  #define SLEEF_SINHD8(v)    Sleef_sinhd8_u35avx512f(v)
  #define SLEEF_COSHD8(v)    Sleef_coshd8_u35avx512f(v)
  #define SLEEF_TANHD8(v)    Sleef_tanhd8_u35avx512f(v)
  #define SLEEF_ASINHD8(v)   Sleef_asinhd8_u35avx512f(v)
  #define SLEEF_ACOSHD8(v)   Sleef_acoshd8_u35avx512f(v)
  #define SLEEF_ATANHD8(v)   Sleef_atanhd8_u35avx512f(v)

  #define SLEEF_SINF8(v)     Sleef_sinf8_u35avx2(v)
  #define SLEEF_COSF8(v)     Sleef_cosf8_u35avx2(v)
  #define SLEEF_TANF8(v)     Sleef_tanf8_u35avx2(v)
  #define SLEEF_ASINF8(v)    Sleef_asinf8_u35avx2(v)
  #define SLEEF_ACOSF8(v)    Sleef_acosf8_u35avx2(v)
  #define SLEEF_ATANF8(v)    Sleef_atanf8_u35avx2(v)
  #define SLEEF_SINHF8(v)    Sleef_sinhf8_u35avx2(v)
  #define SLEEF_COSHF8(v)    Sleef_coshf8_u35avx2(v)
  #define SLEEF_TANHF8(v)    Sleef_tanhf8_u35avx2(v)
  #define SLEEF_ASINHF8(v)   Sleef_asinhf8_u35avx2(v)
  #define SLEEF_ACOSHF8(v)   Sleef_acoshf8_u35avx2(v)
  #define SLEEF_ATANHF8(v)   Sleef_atanhf8_u35avx2(v)

  #define SLEEF_SIND4(v)     Sleef_sind4_u35avx2(v)
  #define SLEEF_COSD4(v)     Sleef_cosd4_u35avx2(v)
  #define SLEEF_TAND4(v)     Sleef_tand4_u35avx2(v)
  #define SLEEF_ASIND4(v)    Sleef_asind4_u35avx2(v)
  #define SLEEF_ACOSD4(v)    Sleef_acosd4_u35avx2(v)
  #define SLEEF_ATAND4(v)    Sleef_atand4_u35avx2(v)
  #define SLEEF_SINHD4(v)    Sleef_sinhd4_u35avx2(v)
  #define SLEEF_COSHD4(v)    Sleef_coshd4_u35avx2(v)
  #define SLEEF_TANHD4(v)    Sleef_tanhd4_u35avx2(v)
  #define SLEEF_ASINHD4(v)   Sleef_asinhd4_u35avx2(v)
  #define SLEEF_ACOSHD4(v)   Sleef_acoshd4_u35avx2(v)
  #define SLEEF_ATANHD4(v)   Sleef_atanhd4_u35avx2(v)
#else
  // u10 (default)
  #define SLEEF_SINF16(v)    Sleef_sinf16_u10avx512f(v)
  #define SLEEF_COSF16(v)    Sleef_cosf16_u10avx512f(v)
  #define SLEEF_TANF16(v)    Sleef_tanf16_u10avx512f(v)
  #define SLEEF_ASINF16(v)   Sleef_asinf16_u10avx512f(v)
  #define SLEEF_ACOSF16(v)   Sleef_acosf16_u10avx512f(v)
  #define SLEEF_ATANF16(v)   Sleef_atanf16_u10avx512f(v)
  #define SLEEF_SINHF16(v)   Sleef_sinhf16_u10avx512f(v)
  #define SLEEF_COSHF16(v)   Sleef_coshf16_u10avx512f(v)
  #define SLEEF_TANHF16(v)   Sleef_tanhf16_u10avx512f(v)
  #define SLEEF_ASINHF16(v)  Sleef_asinhf16_u10avx512f(v)
  #define SLEEF_ACOSHF16(v)  Sleef_acoshf16_u10avx512f(v)
  #define SLEEF_ATANHF16(v)  Sleef_atanhf16_u10avx512f(v)

  #define SLEEF_SIND8(v)     Sleef_sind8_u10avx512f(v)
  #define SLEEF_COSD8(v)     Sleef_cosd8_u10avx512f(v)
  #define SLEEF_TAND8(v)     Sleef_tand8_u10avx512f(v)
  #define SLEEF_ASIND8(v)    Sleef_asind8_u10avx512f(v)
  #define SLEEF_ACOSD8(v)    Sleef_acosd8_u10avx512f(v)
  #define SLEEF_ATAND8(v)    Sleef_atand8_u10avx512f(v)
  #define SLEEF_SINHD8(v)    Sleef_sinhd8_u10avx512f(v)
  #define SLEEF_COSHD8(v)    Sleef_coshd8_u10avx512f(v)
  #define SLEEF_TANHD8(v)    Sleef_tanhd8_u10avx512f(v)
  #define SLEEF_ASINHD8(v)   Sleef_asinhd8_u10avx512f(v)
  #define SLEEF_ACOSHD8(v)   Sleef_acoshd8_u10avx512f(v)
  #define SLEEF_ATANHD8(v)   Sleef_atanhd8_u10avx512f(v)

  #define SLEEF_SINF8(v)     Sleef_sinf8_u10avx2(v)
  #define SLEEF_COSF8(v)     Sleef_cosf8_u10avx2(v)
  #define SLEEF_TANF8(v)     Sleef_tanf8_u10avx2(v)
  #define SLEEF_ASINF8(v)    Sleef_asinf8_u10avx2(v)
  #define SLEEF_ACOSF8(v)    Sleef_acosf8_u10avx2(v)
  #define SLEEF_ATANF8(v)    Sleef_atanf8_u10avx2(v)
  #define SLEEF_SINHF8(v)    Sleef_sinhf8_u10avx2(v)
  #define SLEEF_COSHF8(v)    Sleef_coshf8_u10avx2(v)
  #define SLEEF_TANHF8(v)    Sleef_tanhf8_u10avx2(v)
  #define SLEEF_ASINHF8(v)   Sleef_asinhf8_u10avx2(v)
  #define SLEEF_ACOSHF8(v)   Sleef_acoshf8_u10avx2(v)
  #define SLEEF_ATANHF8(v)   Sleef_atanhf8_u10avx2(v)

  #define SLEEF_SIND4(v)     Sleef_sind4_u10avx2(v)
  #define SLEEF_COSD4(v)     Sleef_cosd4_u10avx2(v)
  #define SLEEF_TAND4(v)     Sleef_tand4_u10avx2(v)
  #define SLEEF_ASIND4(v)    Sleef_asind4_u10avx2(v)
  #define SLEEF_ACOSD4(v)    Sleef_acosd4_u10avx2(v)
  #define SLEEF_ATAND4(v)    Sleef_atand4_u10avx2(v)
  #define SLEEF_SINHD4(v)    Sleef_sinhd4_u10avx2(v)
  #define SLEEF_COSHD4(v)    Sleef_coshd4_u10avx2(v)
  #define SLEEF_TANHD4(v)    Sleef_tanhd4_u10avx2(v)
  #define SLEEF_ASINHD4(v)   Sleef_asinhd4_u10avx2(v)
  #define SLEEF_ACOSHD4(v)   Sleef_acoshd4_u10avx2(v)
  #define SLEEF_ATANHD4(v)   Sleef_atanhd4_u10avx2(v)
#endif

// ---------------- SIMD adapters (only compiled if CPU macro present) -----------
#if defined(__AVX512F__)
template <UnaryKind K> static inline __m512  vf_apply512(__m512 v) {
  if constexpr (K==UnaryKind::Sin)   return SLEEF_SINF16(v);
  if constexpr (K==UnaryKind::Cos)   return SLEEF_COSF16(v);
  if constexpr (K==UnaryKind::Tan)   return SLEEF_TANF16(v);
  if constexpr (K==UnaryKind::Asin)  return SLEEF_ASINF16(v);
  if constexpr (K==UnaryKind::Acos)  return SLEEF_ACOSF16(v);
  if constexpr (K==UnaryKind::Atan)  return SLEEF_ATANF16(v);
  if constexpr (K==UnaryKind::Sinh)  return SLEEF_SINHF16(v);
  if constexpr (K==UnaryKind::Cosh)  return SLEEF_COSHF16(v);
  if constexpr (K==UnaryKind::Tanh)  return SLEEF_TANHF16(v);
  if constexpr (K==UnaryKind::Asinh) return SLEEF_ASINHF16(v);
  if constexpr (K==UnaryKind::Acosh) return SLEEF_ACOSHF16(v);
  if constexpr (K==UnaryKind::Atanh) return SLEEF_ATANHF16(v);
  return v;
}
template <UnaryKind K> static inline __m512d vd_apply512(__m512d v) {
  if constexpr (K==UnaryKind::Sin)   return SLEEF_SIND8(v);
  if constexpr (K==UnaryKind::Cos)   return SLEEF_COSD8(v);
  if constexpr (K==UnaryKind::Tan)   return SLEEF_TAND8(v);
  if constexpr (K==UnaryKind::Asin)  return SLEEF_ASIND8(v);
  if constexpr (K==UnaryKind::Acos)  return SLEEF_ACOSD8(v);
  if constexpr (K==UnaryKind::Atan)  return SLEEF_ATAND8(v);
  if constexpr (K==UnaryKind::Sinh)  return SLEEF_SINHD8(v);
  if constexpr (K==UnaryKind::Cosh)  return SLEEF_COSHD8(v);
  if constexpr (K==UnaryKind::Tanh)  return SLEEF_TANHD8(v);
  if constexpr (K==UnaryKind::Asinh) return SLEEF_ASINHD8(v);
  if constexpr (K==UnaryKind::Acosh) return SLEEF_ACOSHD8(v);
  if constexpr (K==UnaryKind::Atanh) return SLEEF_ATANHD8(v);
  return v;
}
#endif

#if defined(__AVX2__)
template <UnaryKind K> static inline __m256 vf_apply256(__m256 v) {
  if constexpr (K==UnaryKind::Sin)   return SLEEF_SINF8(v);
  if constexpr (K==UnaryKind::Cos)   return SLEEF_COSF8(v);
  if constexpr (K==UnaryKind::Tan)   return SLEEF_TANF8(v);
  if constexpr (K==UnaryKind::Asin)  return SLEEF_ASINF8(v);
  if constexpr (K==UnaryKind::Acos)  return SLEEF_ACOSF8(v);
  if constexpr (K==UnaryKind::Atan)  return SLEEF_ATANF8(v);
  if constexpr (K==UnaryKind::Sinh)  return SLEEF_SINHF8(v);
  if constexpr (K==UnaryKind::Cosh)  return SLEEF_COSHF8(v);
  if constexpr (K==UnaryKind::Tanh)  return SLEEF_TANHF8(v);
  if constexpr (K==UnaryKind::Asinh) return SLEEF_ASINHF8(v);
  if constexpr (K==UnaryKind::Acosh) return SLEEF_ACOSHF8(v);
  if constexpr (K==UnaryKind::Atanh) return SLEEF_ATANHF8(v);
  return v;
}
template <UnaryKind K> static inline __m256d vd_apply256(__m256d v) {
  if constexpr (K==UnaryKind::Sin)   return SLEEF_SIND4(v);
  if constexpr (K==UnaryKind::Cos)   return SLEEF_COSD4(v);
  if constexpr (K==UnaryKind::Tan)   return SLEEF_TAND4(v);
  if constexpr (K==UnaryKind::Asin)  return SLEEF_ASIND4(v);
  if constexpr (K==UnaryKind::Acos)  return SLEEF_ACOSD4(v);
  if constexpr (K==UnaryKind::Atan)  return SLEEF_ATAND4(v);
  if constexpr (K==UnaryKind::Sinh)  return SLEEF_SINHD4(v);
  if constexpr (K==UnaryKind::Cosh)  return SLEEF_COSHD4(v);
  if constexpr (K==UnaryKind::Tanh)  return SLEEF_TANHD4(v);
  if constexpr (K==UnaryKind::Asinh) return SLEEF_ASINHD4(v);
  if constexpr (K==UnaryKind::Acosh) return SLEEF_ACOSHD4(v);
  if constexpr (K==UnaryKind::Atanh) return SLEEF_ATANHD4(v);
  return v;
}
#endif

#if defined(__SSE2__)
template <UnaryKind K> static inline __m128 vf_apply128(__m128 v) {
  // float SSE2 family
  #if defined(TRIG_SLEEF_U35)
    if constexpr (K==UnaryKind::Sin)   return Sleef_sinf4_u35sse2(v);
    if constexpr (K==UnaryKind::Cos)   return Sleef_cosf4_u35sse2(v);
    if constexpr (K==UnaryKind::Tan)   return Sleef_tanf4_u35sse2(v);
    if constexpr (K==UnaryKind::Asin)  return Sleef_asinf4_u35sse2(v);
    if constexpr (K==UnaryKind::Acos)  return Sleef_acosf4_u35sse2(v);
    if constexpr (K==UnaryKind::Atan)  return Sleef_atanf4_u35sse2(v);
    if constexpr (K==UnaryKind::Sinh)  return Sleef_sinhf4_u35sse2(v);
    if constexpr (K==UnaryKind::Cosh)  return Sleef_coshf4_u35sse2(v);
    if constexpr (K==UnaryKind::Tanh)  return Sleef_tanhf4_u35sse2(v);
    if constexpr (K==UnaryKind::Asinh) return Sleef_asinhf4_u35sse2(v);
    if constexpr (K==UnaryKind::Acosh) return Sleef_acoshf4_u35sse2(v);
    if constexpr (K==UnaryKind::Atanh) return Sleef_atanhf4_u35sse2(v);
  #else
    if constexpr (K==UnaryKind::Sin)   return Sleef_sinf4_u10sse2(v);
    if constexpr (K==UnaryKind::Cos)   return Sleef_cosf4_u10sse2(v);
    if constexpr (K==UnaryKind::Tan)   return Sleef_tanf4_u10sse2(v);
    if constexpr (K==UnaryKind::Asin)  return Sleef_asinf4_u10sse2(v);
    if constexpr (K==UnaryKind::Acos)  return Sleef_acosf4_u10sse2(v);
    if constexpr (K==UnaryKind::Atan)  return Sleef_atanf4_u10sse2(v);
    if constexpr (K==UnaryKind::Sinh)  return Sleef_sinhf4_u10sse2(v);
    if constexpr (K==UnaryKind::Cosh)  return Sleef_coshf4_u10sse2(v);
    if constexpr (K==UnaryKind::Tanh)  return Sleef_tanhf4_u10sse2(v);
    if constexpr (K==UnaryKind::Asinh) return Sleef_asinhf4_u10sse2(v);
    if constexpr (K==UnaryKind::Acosh) return Sleef_acoshf4_u10sse2(v);
    if constexpr (K==UnaryKind::Atanh) return Sleef_atanhf4_u10sse2(v);
  #endif
  return v;
}
template <UnaryKind K> static inline __m128d vd_apply128(__m128d v) {
  // double SSE2 family is only u10 in SLEEF
  if constexpr (K==UnaryKind::Sin)   return Sleef_sind2_u10sse2(v);
  if constexpr (K==UnaryKind::Cos)   return Sleef_cosd2_u10sse2(v);
  if constexpr (K==UnaryKind::Tan)   return Sleef_tand2_u10sse2(v);
  if constexpr (K==UnaryKind::Asin)  return Sleef_asind2_u10sse2(v);
  if constexpr (K==UnaryKind::Acos)  return Sleef_acosd2_u10sse2(v);
  if constexpr (K==UnaryKind::Atan)  return Sleef_atand2_u10sse2(v);
  if constexpr (K==UnaryKind::Sinh)  return Sleef_sinhd2_u10sse2(v);
  if constexpr (K==UnaryKind::Cosh)  return Sleef_coshd2_u10sse2(v);
  if constexpr (K==UnaryKind::Tanh)  return Sleef_tanhd2_u10sse2(v);
  if constexpr (K==UnaryKind::Asinh) return Sleef_asinhd2_u10sse2(v);
  if constexpr (K==UnaryKind::Acosh) return Sleef_acoshd2_u10sse2(v);
  if constexpr (K==UnaryKind::Atanh) return Sleef_atanhd2_u10sse2(v);
  return v;
}
#endif // __SSE2__
#endif // HAVE_SLEEF

// ----------------------------------------------------------------------------------
// Vectorized core for a contiguous slice (selects best available path)
// ----------------------------------------------------------------------------------
template <typename T, UnaryKind K>
static inline void apply_block(const T* in, T* out, size_t n) {
  size_t i = 0;

#if HAVE_SLEEF
  // Use SLEEF+SIMD only when both SLEEF and the ISA are available
  #if defined(__AVX512F__)
    if constexpr (std::is_same_v<T,float>) {
      constexpr size_t L = 16;
      for (; i + L <= n; i += L) {
        __m512 v = _mm512_loadu_ps(in + i);
        v = vf_apply512<K>(v);
        _mm512_storeu_ps(out + i, v);
      }
    } else if constexpr (std::is_same_v<T,double>) {
      constexpr size_t L = 8;
      for (; i + L <= n; i += L) {
        __m512d v = _mm512_loadu_pd(in + i);
        v = vd_apply512<K>(v);
        _mm512_storeu_pd(out + i, v);
      }
    }
  #elif defined(__AVX2__)
    if constexpr (std::is_same_v<T,float>) {
      constexpr size_t L = 8;
      for (; i + L <= n; i += L) {
        __m256 v = _mm256_loadu_ps(in + i);
        v = vf_apply256<K>(v);
        _mm256_storeu_ps(out + i, v);
      }
    } else if constexpr (std::is_same_v<T,double>) {
      constexpr size_t L = 4;
      for (; i + L <= n; i += L) {
        __m256d v = _mm256_loadu_pd(in + i);
        v = vd_apply256<K>(v);
        _mm256_storeu_pd(out + i, v);
      }
    }
  #elif defined(__SSE2__)
    if constexpr (std::is_same_v<T,float>) {
      constexpr size_t L = 4;
      for (; i + L <= n; i += L) {
        __m128 v = _mm_loadu_ps(in + i);
        v = vf_apply128<K>(v);
        _mm_storeu_ps(out + i, v);
      }
    } else if constexpr (std::is_same_v<T,double>) {
      constexpr size_t L = 2;
      for (; i + L <= n; i += L) {
        __m128d v = _mm_loadu_pd(in + i);
        v = vd_apply128<K>(v);
        _mm_storeu_pd(out + i, v);
      }
    }
  #endif
#endif // HAVE_SLEEF

  // Scalar tail (and entire path when SLEEF/ISA not available)
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
  for (; i < n; ++i) {
    out[i] = scalar_apply<T,K>(in[i]);
  }
}

// ----------------------------------------------------------------------------------
// OpenMP chunker
// ----------------------------------------------------------------------------------
template <typename F>
static inline void for_blocks(size_t n, size_t grain, const F& fn) {
#if defined(_OPENMP)
  #pragma omp parallel for schedule(static)
  for (long long ss = 0; ss < static_cast<long long>(n); ss += static_cast<long long>(grain)) {
    size_t s = static_cast<size_t>(ss);
    size_t e = (s + grain < n) ? (s + grain) : n;
    fn(s, e);
  }
#else
  size_t s = 0;
  while (s < n) {
    size_t e = (s + grain < n) ? (s + grain) : n;
    fn(s, e);
    s = e;
  }
#endif
}

// ----------------------------------------------------------------------------------
// Public template required by the dispatcher
// ----------------------------------------------------------------------------------
template <typename T, UnaryKind K>
void apply_unary_kernel(const T* in, T* out, size_t n) {
  if (n == 0) return;
  constexpr size_t GRAIN = TRIG_PARALLEL_GRAIN;
  for_blocks(n, GRAIN, [&](size_t s, size_t e){
    apply_block<T, K>(in + s, out + s, e - s);
  });
}

// Explicit instantiations for float/double to ensure linkage across TUs
#define OT_INSTANTIATE(KIND) \
  template void apply_unary_kernel<float , UnaryKind::KIND>(const float* , float* , size_t); \
  template void apply_unary_kernel<double, UnaryKind::KIND>(const double*, double*, size_t);

OT_INSTANTIATE(Sin)   OT_INSTANTIATE(Cos)   OT_INSTANTIATE(Tan)
OT_INSTANTIATE(Asin)  OT_INSTANTIATE(Acos)  OT_INSTANTIATE(Atan)
OT_INSTANTIATE(Sinh)  OT_INSTANTIATE(Cosh)  OT_INSTANTIATE(Tanh)
OT_INSTANTIATE(Asinh) OT_INSTANTIATE(Acosh) OT_INSTANTIATE(Atanh)

#undef OT_INSTANTIATE

} // namespace trig_detail
} // namespace OwnTensor
// End of file