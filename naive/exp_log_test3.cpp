#include <cmath>
#include <omp.h>
#include "../../../include/Tensor.h"
#include "../../../include/UnaryDispatcher.hpp"
#include "../../../include/exp_log_kernels.hpp"

namespace exp_log {
    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel_generic(const T* in, T* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            out[i] = Func(in[i]);
        }
    }

    float expf_wrap(float x) { return expf(x); }
    double exp_wrap(double x) { return std::exp(x); }
    float exp2f_wrap(float x) { return std::exp2f(x); }
    double exp2_wrap(double x) { return std::exp2(x); }

    float logf_wrap(float x) { return logf(x); }
    double log_wrap(double x) { return std::log(x); }
    float log2f_wrap(float x) { return std::log2f(x); }
    double log2_wrap(double x) { return std::log2(x); }
    float log10f_wrap(float x) { return log10f(x); }
    double log10_wrap(double x) { return std::log10(x); }
    
    // Exp
    template void apply_unary_kernel_generic<float, expf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel_generic<double, exp_wrap>(const double* in, double* out, size_t size);
    // Exp2
    template void apply_unary_kernel_generic<float, exp2f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel_generic<double, exp2_wrap>(const double* in, double* out, size_t size);
    // Log
    template void apply_unary_kernel_generic<float, logf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel_generic<double, log_wrap>(const double* in, double* out, size_t size);
    // Log2
    template void apply_unary_kernel_generic<float, log2f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel_generic<double, log2_wrap>(const double* in, double* out, size_t size);
    // Log10
    template void apply_unary_kernel_generic<float, log10f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel_generic<double, log10_wrap>(const double* in, double* out, size_t size);

    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel_generic_inplace(T* data, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            data[i] = Func(data[i]); // In-place modification
        }
    }

    template<typename T, T(*Func)(T)>
    void inplace_dispatcher_wrapper(const T* in, T* out, size_t size) {
        apply_unary_kernel_generic<T, Func>(in, const_cast<T*>(in), size);
    }

    // Exp In-place
    template void inplace_dispatcher_wrapper<float, expf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, exp_wrap>(const double* in, double* out, size_t size);
    // Exp2 In-place
    template void inplace_dispatcher_wrapper<float, exp2f_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, exp2_wrap>(const double* in, double* out, size_t size);
    // Log In-place
    template void inplace_dispatcher_wrapper<float, logf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, log_wrap>(const double* in, double* out, size_t size);
    // Log2 In-place
    template void inplace_dispatcher_wrapper<float, log2f_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, log2_wrap>(const double* in, double* out, size_t size);
    // Log10 In-place
    template void inplace_dispatcher_wrapper<float, log10f_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, log10_wrap>(const double* in, double* out, size_t size);
    
}