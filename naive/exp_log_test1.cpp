// #include <cmath>
// #include <omp.h>
// #include <functional>
// #include "../../../include/Tensor.h"
// #include "../../../include/UnaryDispatcher.hpp"

// namespace exp_log {
//     template <typename T, Unary U>
//     void apply_unary_kernel(const T* in, T* out, size_t size) {
//         #pragma omp parallel for
//         for (size_t i = 0; i < size; ++i) {
//             out[i] = exp(in[i]);
//         }
//     }
//     template void apply_unary_kernel<float, Unary::Exp>(const float* in, float* out, size_t size);
// }

// namespace exp_log {
//     // 1. The Generic Kernel
//     // T(*Func)(T) is the function pointer type
//     template<typename T, T(*Func)(T)> 
//     void apply_unary_kernel(const T* in, T* out, size_t size) {
//         #pragma omp parallel for
//         for (size_t i = 0; i < size; ++i) {
//             out[i] = Func(in[i]); // Calls the function specified in the template
//         }
//     }

//     // 2. The Wrapper (Same as before, using cmath)
//     // wrappers for exp, exp2
//     float expf_wrap(float x) { return expf(x); }
//     double exp_wrap(double x) { return exp(x); }
//     float exp2f_wrap(float x) { return exp2f(x); }
//     double exp2_wrap(double x) { return exp2(x); }

//     // for log, log2, log10
//     float logf_wrap(float x) { return logf(x); }
//     double log_wrap(double x) { return log(x); }
//     float log2f_wrap(float x) { return log2f(x); }
//     double log2_wrap(double x) { return log2(x); }
//     float log10f_wrap(float x) { return log10f(x); }
//     double log10_wrap(double x) { return log10(x); }

//     // 3. The Call Site
//     // The compiler "knows" what Func is at compile time.
//     template void apply_unary_kernel<float, expf_wrap>(const float* in, float* out, size_t size);
// }