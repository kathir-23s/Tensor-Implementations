#include <cmath>
#include <omp.h>
#include <stdexcept>
#include "../../../include/Tensor.h"
#include "../../../include/UnaryDispatcher.hpp"
#include "../../../include/exp_log_kernels.hpp"
#include "../../../include/tesnor_unaryops.hpp"
#include "../../../include/DtypeTraits.h"

namespace exp_log {
    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel(const T* in, T* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            out[i] = Func(in[i]);
        }
    }

    // wrappers
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
    template void apply_unary_kernel<float, expf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, exp_wrap>(const double* in, double* out, size_t size);
    // Exp2
    template void apply_unary_kernel<float, exp2f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, exp2_wrap>(const double* in, double* out, size_t size);
    // Log
    template void apply_unary_kernel<float, logf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, log_wrap>(const double* in, double* out, size_t size);
    // Log2
    template void apply_unary_kernel<float, log2f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, log2_wrap>(const double* in, double* out, size_t size);
    // Log10
    template void apply_unary_kernel<float, log10f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, log10_wrap>(const double* in, double* out, size_t size);

    // inplace
    template<typename T, T(*Func)(T)>
    void inplace_dispatcher_wrapper(const T* in, T* out, size_t size) {
        apply_unary_kernel<T, Func>(in, const_cast<T*>(in), size);
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

    // for promotion kernel
    template<typename T_In, typename T_Out, T_Out(*T_OpFunc)(T_Out)>
    void apply_unary_promotion_kernel(const T_In* in, T_Out* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            T_Out temp_val = static_cast<T_Out>(in[i]);
            out[i] = T_OpFunc(temp_val);
        }
    }
    // promotion types [only Out_of_Place]
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::expf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::expf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::exp_wrap>(const int64_t* in, double* out, size_t size);
    template void apply_unary_promotion_kernel<uint16_t, float, exp_log::expf_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::exp2f_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::exp2f_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::exp2_wrap>(const int64_t* in, double* out, size_t size);
    template void apply_unary_promotion_kernel<uint16_t, float, exp_log::exp2f_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::logf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::logf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::log_wrap>(const int64_t* in, double* out, size_t size);
    template void apply_unary_promotion_kernel<uint16_t, float, exp_log::logf_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::log2f_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::log2f_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::log2_wrap>(const int64_t* in, double* out, size_t size);
    template void apply_unary_promotion_kernel<uint16_t, float, exp_log::log2f_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::log10f_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::log10f_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::log10_wrap>(const int64_t* in, double* out, size_t size);
    template void apply_unary_promotion_kernel<uint16_t, float, exp_log::log10f_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
}
