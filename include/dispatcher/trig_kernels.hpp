#pragma once

#include <cmath>
#include <omp.h>
#include <stdexcept>
#include "../../../include/Tensor.h"
#include "../../../include/dispatcher/UnaryDispatcher.hpp"
#include "../../../include/dispatcher/exp_log_kernels.hpp"
#include "../../../include/dispatcher/tesnor_unaryops.hpp"
#include "../../../include/DtypeTraits.h"
#include "../../../include/types.h"

namespace trig {
    // wrappers
    //trigonometric functics
    float sinf_wrap(float x);
    double sin_wrap(double x);
    float cosf_wrap(float x);
    double cos_wrap(double x);
    float tanf_wrap(float x);
    double tan_wrap(double x);
    // trigonometric arc functions
    float asinf_wrap(float x);
    double asin_wrap(double x);
    float acosf_wrap(float x);
    double acos_wrap(double x);
    float atanf_wrap(float x);
    double atan_wrap(double x);
    // hyperbolic functions
    float sinhf_wrap(float x);
    double sinh_wrap(double x);
    float coshf_wrap(float x);
    double cosh_wrap(double x);
    float tanhf_wrap(float x);
    double tanh_wrap(double x);
    // hyperbolic arc functions
    float asinhf_wrap(float x);
    double asinh_wrap(double x);
    float acoshf_wrap(float x);
    double acosh_wrap(double x);
    float atanhf_wrap(float x);
    double atanh_wrap(double x);

    // outplace
    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel(const T* in, T* out, size_t size); 
    // inplace
    template<typename T, T(*Func)(T)>
    void inplace_dispatcher_wrapper(const T* in, T* out, size_t size);
    // for promotion kernel
    template<typename T_In, typename T_Out, T_Out(*T_OpFunc)(T_Out)>
    void apply_unary_promotion_kernel(const T_In* in, T_Out* out, size_t size);

    // Sin Out-Place
    extern template void apply_unary_kernel<float, sinf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, sin_wrap>(const double* in, double* out, size_t size);
    // Cos Out-Place
    extern template void apply_unary_kernel<float, cosf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, cos_wrap>(const double* in, double* out, size_t size);
    // Tan Out-Place
    extern template void apply_unary_kernel<float, tanf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, tan_wrap>(const double* in, double* out, size_t size);
    // Asin Out-Place
    extern template void apply_unary_kernel<float, asinf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, asin_wrap>(const double* in, double* out, size_t size);
    // Acos Out-Place
    extern template void apply_unary_kernel<float, acosf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, acos_wrap>(const double* in, double* out, size_t size);
    // Atan Out-Place
    extern template void apply_unary_kernel<float, atanf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, atan_wrap>(const double* in, double* out, size_t size);
    // Sinh Out-Place
    extern template void apply_unary_kernel<float, sinhf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, sinh_wrap>(const double* in, double* out, size_t size);
    // Cosh Out-Place
    extern template void apply_unary_kernel<float, coshf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, cosh_wrap>(const double* in, double* out, size_t size);
    // Tanh Out-Place
    extern template void apply_unary_kernel<float, tanhf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, tanh_wrap>(const double* in, double* out, size_t size);
    // Asinh Out-Place
    extern template void apply_unary_kernel<float, asinhf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, asinh_wrap>(const double* in, double* out, size_t size);
    // Acosh Out-Place
    extern template void apply_unary_kernel<float, acoshf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, acosh_wrap>(const double* in, double* out, size_t size);
    // Atanh Out-Place
    extern template void apply_unary_kernel<float, atanhf_wrap>(const float* in, float* out, size_t size);
    extern template void apply_unary_kernel<double, atanh_wrap>(const double* in, double* out, size_t size);

    // Sin In-place
    extern template void inplace_dispatcher_wrapper<float, sinf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, sin_wrap>(const double* in, double* out, size_t size);
    // Cos In-place
    extern template void inplace_dispatcher_wrapper<float, cosf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, cos_wrap>(const double* in, double* out, size_t size);
    // Tan In-place
    extern template void inplace_dispatcher_wrapper<float, tanf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, tan_wrap>(const double* in, double* out, size_t size);
    // Asin In-place
    extern template void inplace_dispatcher_wrapper<float, asinf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, asin_wrap>(const double* in, double* out, size_t size);
    // Acos In-place
    extern template void inplace_dispatcher_wrapper<float, acosf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, acos_wrap>(const double* in, double* out, size_t size);
    // Atan In-place
    extern template void inplace_dispatcher_wrapper<float, atanf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, atan_wrap>(const double* in, double* out, size_t size);
    // Sinh In-place
    extern template void inplace_dispatcher_wrapper<float, sinhf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, sinh_wrap>(const double* in, double* out, size_t size);
    // Cosh In-place
    extern template void inplace_dispatcher_wrapper<float, coshf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, cosh_wrap>(const double* in, double* out, size_t size);
    // Tanh In-place
    extern template void inplace_dispatcher_wrapper<float, tanhf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, tanh_wrap>(const double* in, double* out, size_t size);
    // Asinh In-place
    extern template void inplace_dispatcher_wrapper<float, asinhf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, asinh_wrap>(const double* in, double* out, size_t size);
    // Acosh In-place
    extern template void inplace_dispatcher_wrapper<float, acoshf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, acosh_wrap>(const double* in, double* out, size_t size);
    // Atanh In-place
    extern template void inplace_dispatcher_wrapper<float, atanhf_wrap>(const float* in, float* out, size_t size);
    extern template void inplace_dispatcher_wrapper<double, atanh_wrap>(const double* in, double* out, size_t size);

    // promotion types [only Out_of_Place]
    // Sin Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::sinf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::sinf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::sin_wrap>(const int64_t* in, double* out, size_t size);
    // Cos Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::cosf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::cosf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::cos_wrap>(const int64_t* in, double* out, size_t size);
    // Tan Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::tanf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::tanf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::tan_wrap>(const int64_t* in, double* out, size_t size);
    // Asin Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::asinf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::asinf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::asin_wrap>(const int64_t* in, double* out, size_t size);
    // Acos Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::acosf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::acosf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::acos_wrap>(const int64_t* in, double* out, size_t size);
    // Atan Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::atanf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::atanf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::atan_wrap>(const int64_t* in, double* out, size_t size);
    // Sinh Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::sinhf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::sinhf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::sinh_wrap>(const int64_t* in, double* out, size_t size);
    // Cosh Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::coshf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::coshf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::cosh_wrap>(const int64_t* in, double* out, size_t size);
    // Tanh Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::tanhf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::tanhf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::tanh_wrap>(const int64_t* in, double* out, size_t size);
    // Asinh Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::asinhf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::asinhf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::asinh_wrap>(const int64_t* in, double* out, size_t size);
    // Acosh Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::acoshf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::acoshf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::acosh_wrap>(const int64_t* in, double* out, size_t size);
    // Atanh Promotion
    extern template void apply_unary_promotion_kernel<int16_t, float, trig::atanhf_wrap>(const int16_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int32_t, float, trig::atanhf_wrap>(const int32_t* in, float* out, size_t size);
    extern template void apply_unary_promotion_kernel<int64_t, double, trig::atanh_wrap>(const int64_t* in, double* out, size_t size);
    
}