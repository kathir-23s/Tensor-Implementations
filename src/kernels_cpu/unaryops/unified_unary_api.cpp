#include <cmath>
#include <omp.h>
#include <stdexcept>
#include "../../../include/Tensor.h"
#include "../../../include/UnaryDispatcher.hpp"
#include "../../../include/exp_log_kernels.hpp"
#include "../../../include/tesnor_unaryops.hpp"
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
    // 1. Build the Kernel Key
    KernelKey key{ op, input.dtype(), Device::CPU, mode }; // no logic available toi handle device now.
    
    // 2. Lookup the Kernel
    UnaryKernelFn fn = KernelRegistry::instance().get_kernel(key);

    if (!fn) {
        throw std::runtime_error(
            "Kernel not found for " + unaryop_to_string(op) + " | " + 
            dtype_to_string(input.dtype()) + " | " + 
            (mode == ExecutionMode::In_Place ? "InPlace" : "OutPlace")
        );
    }
    
    // 3. Execute the Kernel
    fn(input.data(), output.data(), input.numel(), param);

    return output;
}

// Helper function to get the promoted Dtype (from your logic)
static Dtype promoted_type_for(const Tensor& t, UnaryOp op) {
    if (op_promotes_int(op) && is_int(t.dtype())) {
        return Dtype::Float32; 
    }
    return t.dtype();
}

// static inline bool is_int(Dtype d) { 
//     return d == Dtype::Int16 || d == Dtype::Int32 || d == Dtype::Int64; 
// }


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

void log_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Log) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Log' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Log, ExecutionMode::In_Place, 0.0);
}

void exp2_(Tensor& input) {
    if (op_promotes_int(UnaryOp::Exp2) && is_int(input.dtype())) {
        throw std::runtime_error("In-place failed: Cannot perform 'Exp2' on integer tensor.");
    }
    _dispatch_unary_op_internal(input, input, UnaryOp::Exp2, ExecutionMode::In_Place, 0.0);
}
// ... continue for log2_ and log10_ ...

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

Tensor exp(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Exp);
    Tensor output({input.shape()}, output_dtype, input.device()); // Allocate
    _dispatch_unary_op_internal(input, output, UnaryOp::Exp, ExecutionMode::Out_of_Place, 0.0);
    return output;
}

Tensor log(const Tensor& input) {
    Dtype output_dtype = promoted_type_for(input, UnaryOp::Log);
    Tensor output({input.shape()}, output_dtype, input.device());
    _dispatch_unary_op_internal(input, output, UnaryOp::Log, ExecutionMode::Out_of_Place, 0.0);
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
