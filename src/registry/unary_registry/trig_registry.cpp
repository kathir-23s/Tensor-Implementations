#include "../include/Tensor.h"
#include "../include/types.h"
#include "../include/dispatcher/UnaryDispatcher.hpp"
#include "../include/dispatcher/UnaryAutoRegister.hpp"
#include "../include/dispatcher/trig_kernels.hpp"
#include "../../include/types.h"

// --- OUT OF PLACE REGISTRATION ---
// Sin Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::apply_unary_kernel<float, trig::sinf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::apply_unary_kernel<double, trig::sin_wrap>));
// Cos Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::apply_unary_kernel<float, trig::cosf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::apply_unary_kernel<double, trig::cos_wrap>));
// Tan Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::apply_unary_kernel<float, trig::tanf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::apply_unary_kernel<double, trig::tan_wrap>));
// Asin Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::apply_unary_kernel<float, trig::asinf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::apply_unary_kernel<double, trig::asin_wrap>));
// Acos Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::apply_unary_kernel<float, trig::acosf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::apply_unary_kernel<double, trig::acos_wrap>));
// Atan Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::apply_unary_kernel<float, trig::atanf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::apply_unary_kernel<double, trig::atan_wrap>));
// Asin Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::apply_unary_kernel<float, trig::sinhf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::apply_unary_kernel<double, trig::sinh_wrap>));
// Acos Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::apply_unary_kernel<float, trig::coshf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::apply_unary_kernel<double, trig::cosh_wrap>));
// Atan Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::apply_unary_kernel<float, trig::tanhf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::apply_unary_kernel<double, trig::tanh_wrap>));
// Asinh Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Asinh, UnaryOp::Asinh, Device::CPU, (&trig::apply_unary_kernel<float, trig::asinhf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Asinh, UnaryOp::Exp, Device::CPU, (&trig::apply_unary_kernel<double, trig::asinh_wrap>));
// AcoshOut-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::apply_unary_kernel<float, trig::acoshf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::apply_unary_kernel<double, trig::acosh_wrap>));
// Atanh Out-Place
REGISTER_UNARY_KERNEL_OP(float, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::apply_unary_kernel<float, trig::atanhf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::apply_unary_kernel<double, trig::atanh_wrap>));

// --- IN PLACE REGISTRATION ---
// Sin Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::sinf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::sin_wrap>));
// Cos Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::cosf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::cos_wrap>));
// Tan Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::tanf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::tan_wrap>));
// Asin Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::asinf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::asin_wrap>));
// Acos Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::acosf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::acos_wrap>));
// Atan Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::atanf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::atan_wrap>));
// Asin Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::sinhf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::sinh_wrap>));
// Acos Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::coshf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::cosh_wrap>));
// Atan Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::tanhf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::tanh_wrap>));
// Asinh Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Asinh, UnaryOp::Asinh, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::asinhf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Asinh, UnaryOp::Exp, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::asinh_wrap>));
// AcoshOut-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::acoshf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::acosh_wrap>));
// Atanh Out-Place
REGISTER_UNARY_KERNEL_IP(float, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::inplace_dispatcher_wrapper<float, trig::atanhf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::inplace_dispatcher_wrapper<double, trig::atanh_wrap>));

// --- PROMOTION REGISTRATIONS ---
// Sin Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::sinf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::sinf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Sin, UnaryOp::Sin, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::sin_wrap>));
// Cos Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::cosf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::cosf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Cos, UnaryOp::Cos, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::cos_wrap>));
// Tan Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::tanf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::tanf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Tan, UnaryOp::Tan, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::tan_wrap>));
// Asin Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::asinf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::asinf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Asin, UnaryOp::Asin, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::asin_wrap>));
// Acos Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::acosf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::acosf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Acos, UnaryOp::Acos, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::acos_wrap>));
// Atan Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::atanf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::atanf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Atan, UnaryOp::Atan, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::atan_wrap>));
// Sinh Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::sinhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::sinhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Sinh, UnaryOp::Sinh, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::sinh_wrap>));
// Cosh Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::coshf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::coshf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Cosh, UnaryOp::Cosh, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::cosh_wrap>));
// Tanh Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::tanhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::tanhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Tanh, UnaryOp::Tanh, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::tanh_wrap>));
// Asin Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Asinh, UnaryOp::Asinh, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::asinhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Asinh, UnaryOp::Asinh, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::asinhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Asinh, UnaryOp::Asinh, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::asinh_wrap>));
// Acos Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::acoshf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::acoshf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Acosh, UnaryOp::Acosh, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::acosh_wrap>));
// Atan Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::apply_unary_promotion_kernel<int16_t, float, trig::atanhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::apply_unary_promotion_kernel<int32_t, float, trig::atanhf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, trig::Unary::Atanh, UnaryOp::Atanh, Device::CPU, (&trig::apply_unary_promotion_kernel<int64_t, double, trig::atanh_wrap>));