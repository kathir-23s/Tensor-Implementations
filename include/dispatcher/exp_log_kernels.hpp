#pragma once
#include <cstddef>
#include <cstdint>
#include "../types.h"

namespace exp_log {
    float expf_wrap(float x);
    double exp_wrap(double x);
    float exp2f_wrap(float x);
    double exp2_wrap(double x);
    float logf_wrap(float x);
    double log_wrap(double x);
    float log2f_wrap(float x);
    double log2_wrap(double x);
    float log10f_wrap(float x);
    double log10_wrap(double x);

    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel(const T* in, T* out, size_t size);

    template<typename T, T(*Func)(T)>
    void inplace_dispatcher_wrapper(const T* in, T* out, size_t size);

    template<typename T_In, typename T_Out, T_Out(*T_OpFunc)(T_Out)>
    void apply_unary_promotion_kernel(const T_In* in, T_Out* out, size_t size);
    
    // Declarations of the explicit instantiations
    // Out_of_Place
    // extern template void apply_unary_kernel<bfloat16_t, expf_wrap>(const bfloat16_t* in, bfloat16_t* out, size_t size);
    extern template void apply_unary_kernel<float, expf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, exp_wrap>(const double* in, double* out, size_t size);
    extern template void apply_unary_kernel<float, exp2f_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, exp2_wrap>(const double* in, double* out, size_t size);
    extern template void apply_unary_kernel<float, logf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, log_wrap>(const double* in, double* out, size_t size);
    extern template void apply_unary_kernel<float, log2f_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, log2_wrap>(const double* in, double* out, size_t size);
    extern template void apply_unary_kernel<float, log10f_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, log10_wrap>(const double* in, double* out, size_t size);
    // In_Place
    extern template void inplace_dispatcher_wrapper<float, expf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, exp_wrap>(const double* in, double* out, size_t size);
    extern template void inplace_dispatcher_wrapper<float, exp2f_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, exp2_wrap>(const double* in, double* out, size_t size);
    extern template void inplace_dispatcher_wrapper<float, logf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, log_wrap>(const double* in, double* out, size_t size);
    extern template void inplace_dispatcher_wrapper<float, log2f_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, log2_wrap>(const double* in, double* out, size_t size);
    extern template void inplace_dispatcher_wrapper<float, log2f_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, log2_wrap>(const double* in, double* out, size_t size);
    // type promotion [only Out_of_place]
    extern template void apply_unary_promotion_kernel<int16_t, float, exp_log::expf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, exp_log::expf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, exp_log::exp_wrap>(const int64_t* in, double* out, size_t size);
    // extern template void apply_unary_promotion_kernel<uint16_t, float, exp_log::expf_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    extern template void apply_unary_promotion_kernel<int16_t, float, exp_log::exp2f_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, exp_log::exp2f_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, exp_log::exp2_wrap>(const int64_t* in, double* out, size_t size);
    // extern template void apply_unary_promotion_kernel<uint16_t, float, exp_log::exp2f_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    extern template void apply_unary_promotion_kernel<int16_t, float, exp_log::logf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, exp_log::logf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, exp_log::log_wrap>(const int64_t* in, double* out, size_t size);
    // extern template void apply_unary_promotion_kernel<uint16_t, float, exp_log::logf_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    extern template void apply_unary_promotion_kernel<int16_t, float, exp_log::log2f_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, exp_log::log2f_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, exp_log::log_wrap>(const int64_t* in, double* out, size_t size);
    // extern template void apply_unary_promotion_kernel<uint16_t, float, exp_log::log2f_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
    extern template void apply_unary_promotion_kernel<int16_t, float, exp_log::log10f_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, exp_log::log10f_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, exp_log::log10_wrap>(const int64_t* in, double* out, size_t size);
    // extern template void apply_unary_promotion_kernel<uint16_t, float, exp_log::log10f_wrap>(const uint16_t* in, float* out, size_t size); // bf16 and f16
}