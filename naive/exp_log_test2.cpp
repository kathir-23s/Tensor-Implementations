// works for outplace
#include <cmath>
#include <omp.h>
#include "../../../include/Tensor.h"
#include "../../../include/UnaryDispatcher.hpp"
#include "../../../include/exp_log_kernels.hpp"

// =========================================================================================
// Generic Kernel Implementation (Template)
// =========================================================================================

namespace exp_log {
    // T(*Func)(T) is the function pointer type for a cmath function like std::expf or std::exp
    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel_generic(const T* in, T* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            out[i] = Func(in[i]); // Calls the function specified in the template argument
        }
    }

    // =====================================================================================
    // Wrapper Function Pointers (For cmath functions that might be overloaded)
    // =====================================================================================
    // We need these explicit wrappers to ensure we get a unique function pointer 
    // for the generic kernel template instantiation.

    // Exponentiation Wrappers
    float expf_wrap(float x) { return expf(x); }
    double exp_wrap(double x) { return std::exp(x); }
    float exp2f_wrap(float x) { return std::exp2f(x); }
    double exp2_wrap(double x) { return std::exp2(x); }

    // Logarithmic Wrappers
    float logf_wrap(float x) { return logf(x); }
    double log_wrap(double x) { return std::log(x); }
    float log2f_wrap(float x) { return std::log2f(x); }
    double log2_wrap(double x) { return std::log2(x); }
    float log10f_wrap(float x) { return log10f(x); }
    double log10_wrap(double x) { return std::log10(x); }

    // =====================================================================================
    // Explicit Template Instantiations (The Call Site)
    // =====================================================================================
    // These create the actual specialized functions that will be registered.
    
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
}