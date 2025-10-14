#include <cmath>
#include <omp.h>
#include <stdexcept>
#include "../../../include/Tensor.h"
#include "../../../include/dispatcher/UnaryDispatcher.hpp"
#include "../../../include/dispatcher/exp_log_kernels.hpp"
#include "../../../include/dispatcher/tesnor_unaryops.hpp"
#include "../../../include/DtypeTraits.h"
#include "../../../include/UnaryOps.h"
// =========================================================================
// INTERNAL UNIFIED DISPATCHER (The only logic that touches the registry)
// =========================================================================
static Tensor& _dispatch_unary_op_internal(
    const Tensor& input, 
    Tensor& output, 
    UnaryOp op, 
    ExecutionMode mode, 
    double param = 0.0
) {
    KernelKey key{ op, input.dtype(), Device::CPU, mode }; // no logic available to handle device now.
    UnaryKernelFn fn = KernelRegistry::instance().get_kernel(key);
    if (!fn) {
        throw std::runtime_error(
            "Kernel not found for " + unaryop_to_string(op) + " | " + 
            dtype_to_string(input.dtype()) + " | " + 
            (mode == ExecutionMode::In_Place ? "InPlace" : "OutPlace")
        );
    }
    fn(input.data(), output.data(), input.numel(), param);
    return output;
}
// Helper function to get the promoted Dtype for a unary operation
static inline Dtype promoted_type_for(const Tensor& t, UnaryOp op) {
    if (op_promotes_int(op) && is_int(t.dtype())) {
        if (is_int_less_than_or_equal_to_32bit(t.dtype())) {
            return Dtype::Float32;
        } else if (t.dtype() == Dtype::Int64) {
            return Dtype::Float64;
        }
    }
    return t.dtype();
}
// =========================================================================
// I. IN-PLACE API IMPLEMENTATIONS (Checks safety, then calls dispatcher)
// =========================================================================
// --- EXPONENTIAL / LOGARITHMIC ---
void exp_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Exp) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Exp' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Exp, ExecutionMode::In_Place, 0.0);
}
void exp2_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Exp2) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Exp2' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Exp2, ExecutionMode::In_Place, 0.0);
}
void log_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Log) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Log' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Log, ExecutionMode::In_Place, 0.0);
}
void log2_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Log2) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Log' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Log2, ExecutionMode::In_Place, 0.0);
}
void log10_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Log10) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Log' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Log10, ExecutionMode::In_Place, 0.0);
}

// --- TRIGONOMETRIC (Example) ---
void sin_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Sin) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Sin' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Sin, ExecutionMode::In_Place, 0.0);
}

// --- ALGEBRAIC (Example: Parametric) ---
void pow_(Tensor& input, double exponent) {
    if (op_promotes_int(UnaryOp::Power) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Power' on integer tensor.");
    }
    // Pass the parameter to the dispatcher
    _dispatch_unary_op_internal(input, input, UnaryOp::Power, ExecutionMode::In_Place, exponent);
}


// =========================================================================
// II. OUT-OF-PLACE API IMPLEMENTATIONS (Handles allocation and calls dispatcher)
// =========================================================================
// --- EXPONENTIAL / LOGARITHMIC ---
Tensor exp(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Exp);
    Tensor output({input.shape()}, output_dtype, input.device()); // Allocate
    _dispatch_unary_op_internal(input, output, UnaryOp::Exp, ExecutionMode::Out_of_Place, 0.0);
    return output;
}
Tensor exp2(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Exp2);
    Tensor output({input.shape()}, output_dtype, input.device()); // Allocate
    _dispatch_unary_op_internal(input, output, UnaryOp::Exp2, ExecutionMode::Out_of_Place, 0.0);
    return output;
}
Tensor log(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Log);
    Tensor output({input.shape()}, output_dtype, input.device());
    _dispatch_unary_op_internal(input, output, UnaryOp::Log, ExecutionMode::Out_of_Place, 0.0);
    return output;
}
Tensor log2(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Log2);
    Tensor output({input.shape()}, output_dtype, input.device());
    _dispatch_unary_op_internal(input, output, UnaryOp::Log2, ExecutionMode::Out_of_Place, 0.0);
    return output;
}
Tensor log10(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Log10);
    Tensor output({input.shape()}, output_dtype, input.device());
    _dispatch_unary_op_internal(input, output, UnaryOp::Log10, ExecutionMode::Out_of_Place, 0.0);
    return output;
}
// ... continue for all other Out-of-Place functions ...

// Example Parametric Out-of-Place
Tensor pow(const Tensor& input, double exponent) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Power);
    Tensor output({input.shape()}, output_dtype, input.device());
    _dispatch_unary_op_internal(input, output, UnaryOp::Power, ExecutionMode::Out_of_Place, exponent);
    return output;
}
