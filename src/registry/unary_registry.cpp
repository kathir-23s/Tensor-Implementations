#include "../include/Tensor.h"
#include "../include/fp16_bf16_convert.h"
#include "../include/UnaryDispatcher.hpp"
#include "../include/UnaryAutoRegister.hpp"
#include "../include/exp_log_kernels.hpp"

static inline void f16ish_to_f32(const uint16_t* in, float* out, size_t size, bool bf16) {
    if (!bf16) {
        for (size_t i = 0; i < size; ++i) { out[i] = float16_to_float(in[i]); }
    } else {
        for (size_t i = 0; i < size; ++i) { out[i] = bfloat16_to_float(in[i]); }
    }
}

static inline void f32_to_f16ish(const float* in, uint16_t* out, size_t size, bool bf16) {
    if (!bf16) {
        for (size_t i = 0; i < size; ++i) { out[i] = float_to_float16(in[i]); }
    } else {
        for (size_t i = 0; i < size; ++i) { out[i] = float_to_bfloat16(in[i]); }
    }
};

// static inline Dtype promoted_type_for(const Tensor& t, UnaryOp op) {
//     if (op_promotes_int(op) && is_int(t.dtype())) {
//         return Dtype::Float32;
//     }
// }
// Helper function to get the promoted Dtype for a unary operation
static inline Dtype promoted_type_for(const Tensor& t, UnaryOp op) {
    // Current logic: If op promotes and input is an integer
    if (op_promotes_int(op) && is_int(t.dtype())) {
        // --- NEW LOGIC: Check Integer Size for Promotion ---
        if (is_int_less_than_or_equal_to_32bit(t.dtype())) {
            // Int16, Int32 -> Float32 (Standard)
            return Dtype::Float32;
        } else if (t.dtype() == Dtype::Int64) {
            // Int64 -> Float64 (Required for precision preservation)
            return Dtype::Float64;
        }
    }
    // Default: return the input Dtype if no promotion is needed/possible
    return t.dtype();
}

// New registrations for Exp/Log (Float32)
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::exp2f_wrap>));
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::logf_wrap>));
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::log2f_wrap>));
REGISTER_UNARY_KERNEL_OP(float, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_kernel<float, exp_log::log10f_wrap>));

// New registrations for Exp/Log (Float64 - Double)
// You need Dtype::Float64 to be defined (assumed to be available via UnaryDispatcher.hpp)
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::exp_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::exp2_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::log_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Log2, UnaryOp::Log2, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::log2_wrap>));
REGISTER_UNARY_KERNEL_OP(double, exp_log::Unary::Log10, UnaryOp::Log10, Device::CPU, (&exp_log::apply_unary_kernel<double, exp_log::log10_wrap>));

// --- NEW IN-PLACE REGISTRATIONS ---

// The '&exp_log::inplace_dispatcher_wrapper<...>' points to the explicit instantiation
// you correctly placed at the end of exp_log_test1.cpp.

// Exp In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::exp_wrap>));

// Exp2 In-Place
REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::exp2f_wrap>));
REGISTER_UNARY_KERNEL_IP(double, exp_log::Unary::Exp2, UnaryOp::Exp2, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<double, exp_log::exp2_wrap>));

// Log In-Place
// REGISTER_UNARY_KERNEL_IP(float, exp_log::Unary::Log, UnaryOp::Log, Device::CPU, (&exp_log::inplace_dispatcher_wrapper<float, exp_log::logf_wrap>));

// type promotion kernels:
// Assuming you have defined the REGISTER_UNARY_KERNEL_PROMOTE macro and AutoRegisterUnaryPromote struct.
REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int32_t, float, exp_log::expf_wrap>));
REGISTER_UNARY_KERNEL_PROMOTE(int64_t, double, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, (&exp_log::apply_unary_promotion_kernel<int64_t, double, exp_log::exp_wrap>));