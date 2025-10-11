#pragma once
#include "UnaryDispatcher.hpp"

#ifndef __COUNTER__
#define __COUNTER__ __LINE__
#endif

// token-paste helpers
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

// 1. Normal, OUT-OF-PLACE
#define REGISTER_UNARY_KERNEL_OP(DTYPE, ENUMVAL, UNARYOP, DEVICE, FUNC_PTR) \
    static AutoRegisterUnary<DTYPE, decltype(ENUMVAL), ENUMVAL, FUNC_PTR> \
        CONCAT(_auto_reg_op_, __COUNTER__){UNARYOP, DEVICE, ExecutionMode::Out_of_Place}

// 2. Normal, IN-PLACE (For Exp, Log, Sin, etc.)
#define REGISTER_UNARY_KERNEL_IP(DTYPE, ENUMVAL, UNARYOP, DEVICE, FUNC_PTR) \
    static AutoRegisterUnary<DTYPE, decltype(ENUMVAL), ENUMVAL, FUNC_PTR> \
        CONCAT(_auto_reg_ip_, __COUNTER__){UNARYOP, DEVICE, ExecutionMode::In_Place}


// 3. Parametric, OUT-OF-PLACE (Existing functionality)
#define REGISTER_UNARY_KERNEL_PARAM_OP(DTYPE, ENUMVAL, UNARYOP, DEVICE, FUNC_PTR) \
    static AutoRegisterUnaryParam<DTYPE, decltype(ENUMVAL), ENUMVAL, FUNC_PTR> \
        CONCAT(_auto_regp_op_, __COUNTER__){UNARYOP, DEVICE, ExecutionMode::Out_of_Place}

// 4. Parametric, IN-PLACE (For Power, where the input tensor is modified)
#define REGISTER_UNARY_KERNEL_PARAM_IP(DTYPE, ENUMVAL, UNARYOP, DEVICE, FUNC_PTR) \
    static AutoRegisterUnaryParam<DTYPE, decltype(ENUMVAL), ENUMVAL, FUNC_PTR> \
        CONCAT(_auto_regp_ip_, __COUNTER__){UNARYOP, DEVICE, ExecutionMode::In_Place}

// --- Add this to UnaryAutoRegister.hpp ---

// Macro to register a unary kernel with Dtype promotion (InputDtype -> OutputDtype).
// Usage: REGISTER_UNARY_KERNEL_PROMOTE(int32_t, float, exp_log::Unary::Exp, UnaryOp::Exp, Device::CPU, &exp_log::apply_unary_kernel)
#define REGISTER_UNARY_KERNEL_PROMOTE(INPUT_DTYPE, OUTPUT_DTYPE, ENUMVAL, UNARYOP, DEVICE, FUNC_PTR) \
    static AutoRegisterUnaryPromote<INPUT_DTYPE, OUTPUT_DTYPE, decltype(ENUMVAL), ENUMVAL, FUNC_PTR> \
        CONCAT(_auto_reg_promo_, __COUNTER__){UNARYOP, DEVICE}