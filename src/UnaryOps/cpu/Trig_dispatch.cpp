// src/UnaryOps/cpu/Trig_dispatch.cpp
#include <stdexcept>
#include "ops/UnaryOps/Trigonometry.h"
#include "core/Tensor.h"

namespace OwnTensor {

// CPU backends (defined in Trigonometry.cpp)
Tensor sin_cpu  (const Tensor&); void sin__cpu  (Tensor&);
Tensor cos_cpu  (const Tensor&); void cos__cpu  (Tensor&);
Tensor tan_cpu  (const Tensor&); void tan__cpu  (Tensor&);
Tensor asin_cpu (const Tensor&); void asin__cpu (Tensor&);
Tensor acos_cpu (const Tensor&); void acos__cpu (Tensor&);
Tensor atan_cpu (const Tensor&); void atan__cpu (Tensor&);
Tensor sinh_cpu (const Tensor&); void sinh__cpu (Tensor&);
Tensor cosh_cpu (const Tensor&); void cosh__cpu (Tensor&);
Tensor tanh_cpu (const Tensor&); void tanh__cpu (Tensor&);
Tensor asinh_cpu(const Tensor&); void asinh__cpu(Tensor&);
Tensor acosh_cpu(const Tensor&); void acosh__cpu(Tensor&);
Tensor atanh_cpu(const Tensor&); void atanh__cpu(Tensor&);

// CUDA backends: real symbols in GPU builds, stubs otherwise
#ifdef WITH_CUDA
Tensor sin_cuda  (const Tensor&); void sin__cuda  (Tensor&);
Tensor cos_cuda  (const Tensor&); void cos__cuda  (Tensor&);
Tensor tan_cuda  (const Tensor&); void tan__cuda  (Tensor&);
Tensor asin_cuda (const Tensor&); void asin__cuda (Tensor&);
Tensor acos_cuda (const Tensor&); void acos__cuda (Tensor&);
Tensor atan_cuda (const Tensor&); void atan__cuda (Tensor&);
Tensor sinh_cuda (const Tensor&); void sinh__cuda (Tensor&);
Tensor cosh_cuda (const Tensor&); void cosh__cuda (Tensor&);
Tensor tanh_cuda (const Tensor&); void tanh__cuda (Tensor&);
Tensor asinh_cuda(const Tensor&); void asinh__cuda(Tensor&);
Tensor acosh_cuda(const Tensor&); void acosh__cuda(Tensor&);
Tensor atanh_cuda(const Tensor&); void atanh__cuda(Tensor&);
#else
static inline Tensor __stub_out() { throw std::runtime_error("This binary was built without CUDA support."); }
static inline void   __stub_in()  { throw std::runtime_error("This binary was built without CUDA support."); }
inline Tensor sin_cuda  (const Tensor&) { return __stub_out(); } inline void sin__cuda  (Tensor&) { __stub_in(); }
inline Tensor cos_cuda  (const Tensor&) { return __stub_out(); } inline void cos__cuda  (Tensor&) { __stub_in(); }
inline Tensor tan_cuda  (const Tensor&) { return __stub_out(); } inline void tan__cuda  (Tensor&) { __stub_in(); }
inline Tensor asin_cuda (const Tensor&) { return __stub_out(); } inline void asin__cuda (Tensor&) { __stub_in(); }
inline Tensor acos_cuda (const Tensor&) { return __stub_out(); } inline void acos__cuda (Tensor&) { __stub_in(); }
inline Tensor atan_cuda (const Tensor&) { return __stub_out(); } inline void atan__cuda (Tensor&) { __stub_in(); }
inline Tensor sinh_cuda (const Tensor&) { return __stub_out(); } inline void sinh__cuda (Tensor&) { __stub_in(); }
inline Tensor cosh_cuda (const Tensor&) { return __stub_out(); } inline void cosh__cuda (Tensor&) { __stub_in(); }
inline Tensor tanh_cuda (const Tensor&) { return __stub_out(); } inline void tanh__cuda (Tensor&) { __stub_in(); }
inline Tensor asinh_cuda(const Tensor&) { return __stub_out(); } inline void asinh__cuda(Tensor&) { __stub_in(); }
inline Tensor acosh_cuda(const Tensor&) { return __stub_out(); } inline void acosh__cuda(Tensor&) { __stub_in(); }
inline Tensor atanh_cuda(const Tensor&) { return __stub_out(); } inline void atanh__cuda(Tensor&) { __stub_in(); }
#endif

// templated helpers
template <auto CpuFn, auto CudaFn, typename... Args>
static inline decltype(auto) dispatch_out(const Tensor& x, Args&&... args) {
    if (x.device().is_cuda()) return CudaFn(x, std::forward<Args>(args)...);
    return CpuFn(x, std::forward<Args>(args)...);
}
template <auto CpuFn, auto CudaFn, typename... Args>
static inline void dispatch_inplace(Tensor& x, Args&&... args) {
    if (x.device().is_cuda()) { CudaFn(x, std::forward<Args>(args)...); }
    else                      { CpuFn (x, std::forward<Args>(args)...); }
}

// Out-of-place
Tensor sin   (const Tensor& x){ return dispatch_out<&sin_cpu,   &sin_cuda  >(x); }
Tensor cos   (const Tensor& x){ return dispatch_out<&cos_cpu,   &cos_cuda  >(x); }
Tensor tan   (const Tensor& x){ return dispatch_out<&tan_cpu,   &tan_cuda  >(x); }
Tensor asin  (const Tensor& x){ return dispatch_out<&asin_cpu,  &asin_cuda >(x); }
Tensor acos  (const Tensor& x){ return dispatch_out<&acos_cpu,  &acos_cuda >(x); }
Tensor atan  (const Tensor& x){ return dispatch_out<&atan_cpu,  &atan_cuda >(x); }
Tensor sinh  (const Tensor& x){ return dispatch_out<&sinh_cpu,  &sinh_cuda >(x); }
Tensor cosh  (const Tensor& x){ return dispatch_out<&cosh_cpu,  &cosh_cuda >(x); }
Tensor tanh  (const Tensor& x){ return dispatch_out<&tanh_cpu,  &tanh_cuda >(x); }
Tensor asinh (const Tensor& x){ return dispatch_out<&asinh_cpu, &asinh_cuda>(x); }
Tensor acosh (const Tensor& x){ return dispatch_out<&acosh_cpu, &acosh_cuda>(x); }
Tensor atanh (const Tensor& x){ return dispatch_out<&atanh_cpu, &atanh_cuda>(x); }

// In-place
void sin_   (Tensor& x){ dispatch_inplace<&sin__cpu,   &sin__cuda  >(x); }
void cos_   (Tensor& x){ dispatch_inplace<&cos__cpu,   &cos__cuda  >(x); }
void tan_   (Tensor& x){ dispatch_inplace<&tan__cpu,   &tan__cuda  >(x); }
void asin_  (Tensor& x){ dispatch_inplace<&asin__cpu,  &asin__cuda >(x); }
void acos_  (Tensor& x){ dispatch_inplace<&acos__cpu,  &acos__cuda >(x); }
void atan_  (Tensor& x){ dispatch_inplace<&atan__cpu,  &atan__cuda >(x); }
void sinh_  (Tensor& x){ dispatch_inplace<&sinh__cpu,  &sinh__cuda >(x); }
void cosh_  (Tensor& x){ dispatch_inplace<&cosh__cpu,  &cosh__cuda >(x); }
void tanh_  (Tensor& x){ dispatch_inplace<&tanh__cpu,  &tanh__cuda >(x); }
void asinh_ (Tensor& x){ dispatch_inplace<&asinh__cpu, &asinh__cuda>(x); }
void acosh_ (Tensor& x){ dispatch_inplace<&acosh__cpu, &acosh__cuda>(x); }
void atanh_ (Tensor& x){ dispatch_inplace<&atanh__cpu, &atanh__cuda>(x); }

} // namespace OwnTensor
