// Algebraic.cpp
// In-place and out-of-place algebraic kernels (int/float/double).
// Uses a single PARALLEL_FOR macro to keep code DRY.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(_OPENMP)
  #include <omp.h>
  #define PARALLEL_FOR(n) _Pragma("omp parallel for if(n > 4096)")
#else
  #define PARALLEL_FOR(n)
#endif

namespace OwnTensor {
namespace alg_detail {

// -------- helpers --------
template <typename T>
inline T my_abs(T x){
  if constexpr (std::is_floating_point<T>::value) { using std::fabs; return fabs(x); }
  else { return x >= T(0) ? x : T(-x); }
}
template <typename T>
inline T my_sqrt(T x){
  if constexpr (std::is_floating_point<T>::value) { using std::sqrt; return sqrt(x); }
  else { // integer sqrt -> floor(sqrt(x)) for non-negative
    if (x <= T(0)) return T(0);
    T r = static_cast<T>(std::sqrt(static_cast<long double>(x)));
    return r;
  }
}
template <typename T>
inline T my_recip(T x){
  if constexpr (std::is_floating_point<T>::value) return T(1) / x;
  // integer reciprocal: truncating semantics (caller decides if sensible)
  return x == T(0) ? T(0) : T(1) / x;
}
template <typename T> inline T my_sign(T x){ return (x > T(0)) ? T(1) : (x < T(0) ? T(-1) : T(0)); }

// -------- in-place --------
template <typename T> void square_impl(T* p, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) p[i] = p[i]*p[i];
}
template <typename T> void sqrt_impl(T* p, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) p[i] = my_sqrt(p[i]);
}
template <typename T> void negate_impl(T* p, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) p[i] = -p[i];
}
template <typename T> void abs_impl(T* p, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) p[i] = my_abs(p[i]);
}
template <typename T> void sign_impl(T* p, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) p[i] = my_sign(p[i]);
}
template <typename T> void reciprocal_impl(T* p, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) p[i] = my_recip(p[i]);
}

// -------- out-of-place --------
template <typename T> void square2_impl(const T* a, T* b, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) b[i] = a[i]*a[i];
}
template <typename T> void sqrt2_impl(const T* a, T* b, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) b[i] = my_sqrt(a[i]);
}
template <typename T> void negate2_impl(const T* a, T* b, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) b[i] = -a[i];
}
template <typename T> void abs2_impl(const T* a, T* b, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) b[i] = my_abs(a[i]);
}
template <typename T> void sign2_impl(const T* a, T* b, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) b[i] = my_sign(a[i]);
}
template <typename T> void reciprocal2_impl(const T* a, T* b, size_t n){
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i) b[i] = my_recip(a[i]);
}

// -------- parametric power --------
template <typename T> void power_impl(T* p, size_t n, double e){
  const T expv = static_cast<T>(e);
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i){
    if constexpr (std::is_floating_point<T>::value) { using std::pow; p[i] = pow(p[i], expv); }
    else { using std::pow; p[i] = static_cast<T>(pow(static_cast<double>(p[i]), e)); }
  }
}
template <typename T> void power2_impl(const T* a, T* b, size_t n, double e){
  const T expv = static_cast<T>(e);
  PARALLEL_FOR(n)
  for (ptrdiff_t i=0;i<(ptrdiff_t)n;++i){
    if constexpr (std::is_floating_point<T>::value) { using std::pow; b[i] = pow(a[i], expv); }
    else { using std::pow; b[i] = static_cast<T>(pow(static_cast<double>(a[i]), e)); }
  }
}

// ------ explicit instantiations ------
#define INST_T(Ty) \
  template void square_impl<Ty>(Ty*, size_t); \
  template void sqrt_impl<Ty>(Ty*, size_t); \
  template void negate_impl<Ty>(Ty*, size_t); \
  template void abs_impl<Ty>(Ty*, size_t); \
  template void sign_impl<Ty>(Ty*, size_t); \
  template void reciprocal_impl<Ty>(Ty*, size_t); \
  template void square2_impl<Ty>(const Ty*, Ty*, size_t); \
  template void sqrt2_impl<Ty>(const Ty*, Ty*, size_t); \
  template void negate2_impl<Ty>(const Ty*, Ty*, size_t); \
  template void abs2_impl<Ty>(const Ty*, Ty*, size_t); \
  template void sign2_impl<Ty>(const Ty*, Ty*, size_t); \
  template void reciprocal2_impl<Ty>(const Ty*, Ty*, size_t); \
  template void power_impl<Ty>(Ty*, size_t, double); \
  template void power2_impl<Ty>(const Ty*, Ty*, size_t, double);

INST_T(float)  INST_T(double)
INST_T(int16_t) INST_T(int32_t) INST_T(int64_t)
#undef INST_T

} // namespace alg_detail
}