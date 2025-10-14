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
    float sinf_wrap(float x) { return sinf(x); }
    double sin_wrap(double x) { return sin(x); }
    float cosf_wrap(float x) { return cosf(x); }
    double cos_wrap(double x) { return cos(x); }
    float tanf_wrap(float x) { return tanf(x); }
    double tan_wrap(double x) { return tan(x); }
    // trigonometric arc functions
    float asinf_wrap(float x) { return asinf(x); }
    double asin_wrap(double x) { return asin(x); }
    float acosf_wrap(float x) { return acosf(x); }
    double acos_wrap(double x) { return acos(x); }
    float atanf_wrap(float x) { return atanf(x); }
    double atan_wrap(double x) { return atan(x); }
    // hyperbolic functions
    float sinhf_wrap(float x) { return sinhf(x); }
    double sinh_wrap(double x) { return sinh(x); }
    float coshf_wrap(float x) { return coshf(x); }
    double cosh_wrap(double x) { return cosh(x); }
    float tanhf_wrap(float x) { return tanhf(x); }
    double tanh_wrap(double x) { return tanh(x); }
    // hyperbolic arc functions
    float asinhf_wrap(float x) { return asinhf(x); }
    double asinh_wrap(double x) { return asinh(x); }
    float acoshf_wrap(float x) { return acoshf(x); }
    double acosh_wrap(double x) { return acosh(x); }
    float atanhf_wrap(float x) { return atanhf(x); }
    double atanh_wrap(double x) { return atanh(x); }

    // outplace
    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel(const T* in, T* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            out[i] = Func(in[i]);
        }
    }
    // Sin Out-Place
    template void apply_unary_kernel<float, sinf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, sin_wrap>(const double* in, double* out, size_t size);
    // Cos Out-Place
    template void apply_unary_kernel<float, cosf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, cos_wrap>(const double* in, double* out, size_t size);
    // Tan Out-Place
    template void apply_unary_kernel<float, tanf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, tan_wrap>(const double* in, double* out, size_t size);
    // Asin Out-Place
    template void apply_unary_kernel<float, asinf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, asin_wrap>(const double* in, double* out, size_t size);
    // Acos Out-Place
    template void apply_unary_kernel<float, acosf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, acos_wrap>(const double* in, double* out, size_t size);
    // Atan Out-Place
    template void apply_unary_kernel<float, atanf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, atan_wrap>(const double* in, double* out, size_t size);
    // Sinh Out-Place
    template void apply_unary_kernel<float, sinhf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, sinh_wrap>(const double* in, double* out, size_t size);
    // Cosh Out-Place
    template void apply_unary_kernel<float, coshf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, cosh_wrap>(const double* in, double* out, size_t size);
    // Tanh Out-Place
    template void apply_unary_kernel<float, tanhf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, tanh_wrap>(const double* in, double* out, size_t size);
    // Asinh Out-Place
    template void apply_unary_kernel<float, asinhf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, asinh_wrap>(const double* in, double* out, size_t size);
    // Acosh Out-Place
    template void apply_unary_kernel<float, acoshf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, acosh_wrap>(const double* in, double* out, size_t size);
    // Atanh Out-Place
    template void apply_unary_kernel<float, atanhf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, atanh_wrap>(const double* in, double* out, size_t size);

    // inplace
    template<typename T, T(*Func)(T)>
    void inplace_dispatcher_wrapper(const T* in, T* out, size_t size) {
        apply_unary_kernel<T, Func>(in, const_cast<T*>(in), size);
    }
    // Sin In-place
    template void inplace_dispatcher_wrapper<float, sinf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, sin_wrap>(const double* in, double* out, size_t size);
    // Cos In-place
    template void inplace_dispatcher_wrapper<float, cosf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, cos_wrap>(const double* in, double* out, size_t size);
    // Tan In-place
    template void inplace_dispatcher_wrapper<float, tanf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, tan_wrap>(const double* in, double* out, size_t size);
    // Asin In-place
    template void inplace_dispatcher_wrapper<float, asinf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, asin_wrap>(const double* in, double* out, size_t size);
    // Acos In-place
    template void inplace_dispatcher_wrapper<float, acosf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, acos_wrap>(const double* in, double* out, size_t size);
    // Atan In-place
    template void inplace_dispatcher_wrapper<float, atanf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, atan_wrap>(const double* in, double* out, size_t size);
    // Sinh In-place
    template void inplace_dispatcher_wrapper<float, sinhf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, sinh_wrap>(const double* in, double* out, size_t size);
    // Cosh In-place
    template void inplace_dispatcher_wrapper<float, coshf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, cosh_wrap>(const double* in, double* out, size_t size);
    // Tanh In-place
    template void inplace_dispatcher_wrapper<float, tanhf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, tanh_wrap>(const double* in, double* out, size_t size);
    // Asinh In-place
    template void inplace_dispatcher_wrapper<float, asinhf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, asinh_wrap>(const double* in, double* out, size_t size);
    // Acosh In-place
    template void inplace_dispatcher_wrapper<float, acoshf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, acosh_wrap>(const double* in, double* out, size_t size);
    // Atanh In-place
    template void inplace_dispatcher_wrapper<float, atanhf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, atanh_wrap>(const double* in, double* out, size_t size);

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
    // Sin Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::sinf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::sinf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::sin_wrap>(const int64_t* in, double* out, size_t size);
    // Cos Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::cosf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::cosf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::cos_wrap>(const int64_t* in, double* out, size_t size);
    // Tan Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::tanf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::tanf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::tan_wrap>(const int64_t* in, double* out, size_t size);
    // Asin Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::asinf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::asinf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::asin_wrap>(const int64_t* in, double* out, size_t size);
    // Acos Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::acosf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::acosf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::acos_wrap>(const int64_t* in, double* out, size_t size);
    // Atan Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::atanf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::atanf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::atan_wrap>(const int64_t* in, double* out, size_t size);
    // Sinh Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::sinhf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::sinhf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::sinh_wrap>(const int64_t* in, double* out, size_t size);
    // Cosh Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::coshf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::coshf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::cosh_wrap>(const int64_t* in, double* out, size_t size);
    // Tanh Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::tanhf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::tanhf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::tanh_wrap>(const int64_t* in, double* out, size_t size);
    // Asinh Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::asinhf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::asinhf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::asinh_wrap>(const int64_t* in, double* out, size_t size);
    // Acosh Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::acoshf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::acoshf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::acosh_wrap>(const int64_t* in, double* out, size_t size);
    // Atanh Promotion
    template void apply_unary_promotion_kernel<int16_t, float, trig::atanhf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, trig::atanhf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, trig::atanh_wrap>(const int64_t* in, double* out, size_t size);
    
}