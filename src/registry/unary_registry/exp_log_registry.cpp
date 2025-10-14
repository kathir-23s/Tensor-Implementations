#include "../include/Tensor.h"
#include "../include/types.h"
#include "../include/dispatcher/UnaryDispatcher.hpp"
#include "../include/dispatcher/UnaryAutoRegister.hpp"
#include "../include/dispatcher/exp_log_kernels.hpp"
#include "../../include/types.h"

// --- OUT OF PLACE REGISTRATION ---
// Exp Out-Place
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::exp_wrap>));
// Exp2 Out-Place
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::exp2f_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::exp2_wrap>));
// Log Out-Place
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::logf_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::log_wrap>));
// Log2 Out-Place
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::log2f_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::log2_wrap>));
// Log10 Out-Place
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::log10f_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::log10_wrap>));

// --- IN-PLACE REGISTRATIONS ---
// Exp In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::exp_wrap>));
// Exp2 In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::exp2f_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::exp2_wrap>));
// Log In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::logf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::log_wrap>));
// Log2 In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::log2f_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::log2_wrap>));
// Log10 In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::log10f_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::log10_wrap>));

// --- PROMOTION REGISTRATIONS ---
// Exp Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int16_t, float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int32_t, float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int64_t, double, exp_log::exp_wrap>));
// Exp2 Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int16_t, float, exp_log::exp2f_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int32_t, float, exp_log::exp2f_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int64_t, double, exp_log::exp2_wrap>));
// Log Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int16_t, float, exp_log::logf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int32_t, float, exp_log::logf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int64_t, double, exp_log::log_wrap>));
// Log2 Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int16_t, float, exp_log::log2f_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int32_t, float, exp_log::log2f_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int64_t, double, exp_log::log2_wrap>));
// Log10 Promotion
REGISTER_UNARY_KERNEL_PROMOTE(int16_t, float, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int16_t, float, exp_log::log10f_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int32_t, float, exp_log::log10f_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int64_t, double, exp_log::log10_wrap>));

// --- BFLOAT16 REGISTRATION (No Promotion) ---
// REGISTER_UNARY_KERNEL(
//     uint16_t,
//     exp_log::Unary::Exp, 
//     UnaryOp::Exp, 
//     Device::CPU, 
//     // The fully specialized function pointer for Bfloat16 Exp
//     (&exp_log::apply_unary_f16ish_kernel<
//         uint16_t, 
//         bfloat16_to_float, 
//         float_to_bfloat16, 
//         exp_log::expf_wrap
//     >)
// );

// // FLOAT16 Registration (No Promotion)
// REGISTER_UNARY_KERNEL(
//     uint16_t, // Use uint16_t as the data type
//     exp_log::Unary::Exp, 
//     UnaryOp::Exp, 
//     Device::CPU, 
//     // The fully specialized function pointer for Float16 Exp
//     (&exp_log::apply_unary_f16ish_kernel<
//         uint16_t, 
//         float16_to_float, 
//         float_to_float16, 
//         exp_log::expf_wrap
//     >)
// );
// NOTE: Repeat this for all other unary ops (Log, Exp2, etc.)