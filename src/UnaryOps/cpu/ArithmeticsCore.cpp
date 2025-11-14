#include <cmath>
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/arith.hpp"
#include "dtype/DtypeCastUtils.h"

namespace OwnTensor {

// ============================================================================
// Generic CPU Kernel
// ============================================================================

template<typename T_In, typename T_Out, typename Func>
void unary_kernel_cpu(const T_In* in, T_Out* out, size_t size, Func op) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<T_Out>(op(static_cast<T_Out>(in[i])));
    }
}

// ============================================================================
// Generic Out-of-Place CPU Implementation
// ============================================================================

template<typename FloatFunc, typename DoubleFunc>
Tensor generic_unary_out_cpu(const Tensor& input_tensor, Dtype output_dtype, 
                             FloatFunc float_op, DoubleFunc double_op) {
    // Handle bf16/f16 by promoting to float32
    if (input_tensor.dtype() == Dtype::Bfloat16 || input_tensor.dtype() == Dtype::Float16) {
        Tensor temp = convert_half_to_float32(input_tensor);
        Tensor result = generic_unary_out_cpu(temp, Dtype::Float32, float_op, double_op);
        Tensor output(input_tensor.shape(), input_tensor.dtype(), input_tensor.device(), input_tensor.requires_grad());
        convert_float32_to_half(result, output);
        return output;
    }
    
    Tensor output(input_tensor.shape(), output_dtype, input_tensor.device(), input_tensor.requires_grad());
    
    dispatch_by_dtype(input_tensor.dtype(), [&](auto in_type_instance) {
        using T_In = decltype(in_type_instance);
        const T_In* in_ptr = input_tensor.data<T_In>();
        
        dispatch_by_dtype(output_dtype, [&](auto out_type_instance) {
            using T_Out = decltype(out_type_instance);
            T_Out* out_ptr = output.data<T_Out>();
            
            if constexpr (std::is_same_v<T_Out, double>) {
                unary_kernel_cpu(in_ptr, out_ptr, input_tensor.numel(), double_op);
            } else {
                unary_kernel_cpu(in_ptr, out_ptr, input_tensor.numel(), float_op);
            }
        });
    });
    
    return output;
}

// ============================================================================
// Generic In-Place CPU Implementation
// ============================================================================

template<typename FloatFunc, typename DoubleFunc>
void generic_unary_in_cpu(Tensor& input_tensor, FloatFunc float_op, DoubleFunc double_op) {
    // Handle bf16/f16 by promoting to float32
    if (input_tensor.dtype() == Dtype::Bfloat16 || input_tensor.dtype() == Dtype::Float16) {
        Tensor temp = convert_half_to_float32(input_tensor);
        generic_unary_in_cpu(temp, float_op, double_op);
        convert_float32_to_half(temp, input_tensor);
        return;
    }
    
    dispatch_by_dtype(input_tensor.dtype(), [&](auto type_instance) {
        using T = decltype(type_instance);
        T* ptr = input_tensor.data<T>();
        size_t size = input_tensor.numel();
        
        if constexpr (std::is_same_v<T, double>) {
            #pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = double_op(ptr[i]);
            }
        } else {
            #pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<T>(float_op(static_cast<float>(ptr[i])));
            }
        }
    });
}

// ============================================================================
// SQUARE
// ============================================================================

Tensor square_out_cpu_wrap(const Tensor& input_tensor) {
    auto float_fn = [](float x) { return x * x; };
    auto double_fn = [](double x) { return x * x; };
    return generic_unary_out_cpu(input_tensor, get_promoted_dtype_square(input_tensor.dtype()), float_fn, double_fn);
}

void square_in_cpu_wrap(Tensor& input_tensor) {
    auto float_fn = [](float x) { return x * x; };
    auto double_fn = [](double x) { return x * x; };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// ============================================================================
// SQUARE ROOT
// ============================================================================

Tensor square_root_out_cpu_wrap(const Tensor& input_tensor) {
    auto float_fn = [](float x) { return sqrtf(x); };
    auto double_fn = [](double x) { return std::sqrt(x); };
    return generic_unary_out_cpu(input_tensor, get_promoted_dtype(input_tensor.dtype()), float_fn, double_fn);
}

void square_root_in_cpu_wrap(Tensor& input_tensor) {
    if (input_tensor.dtype() == Dtype::Int16 || input_tensor.dtype() == Dtype::Int32 || input_tensor.dtype() == Dtype::Int64) {
        throw std::invalid_argument("In-place square root requires floating point tensor");
    }
    auto float_fn = [](float x) { return sqrtf(x); };
    auto double_fn = [](double x) { return std::sqrt(x); };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// ============================================================================
// RECIPROCAL
// ============================================================================

Tensor reciprocal_out_cpu_wrap(const Tensor& input_tensor) {
    auto float_fn = [](float x) { return 1.0f / x; };
    auto double_fn = [](double x) { return 1.0 / x; };
    return generic_unary_out_cpu(input_tensor, get_promoted_dtype(input_tensor.dtype()), float_fn, double_fn);
}

void reciprocal_in_cpu_wrap(Tensor& input_tensor) {
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"reciprocal\" not implemented for 'Bool'"
        );
    }
    if (input_tensor.dtype() == Dtype::Int16 || input_tensor.dtype() == Dtype::Int32 || input_tensor.dtype() == Dtype::Int64) {
        throw std::invalid_argument("In-place reciprocal requires floating point tensor");
    }
    auto float_fn = [](float x) { return 1.0f / x; };
    auto double_fn = [](double x) { return 1.0 / x; };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// ============================================================================
// NEGATION - Now throws error for Bool tensors (matching PyTorch)
// ============================================================================

Tensor negator_out_cpu_wrap(const Tensor& input_tensor) {
    // PyTorch behavior: neg() not supported for Bool tensors
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
        );
    }
    
    auto float_fn = [](float x) { return -x; };
    auto double_fn = [](double x) { return -x; };
    return generic_unary_out_cpu(input_tensor, input_tensor.dtype(), float_fn, double_fn);
}

void negator_in_cpu_wrap(Tensor& input_tensor) {
    // PyTorch behavior: neg() not supported for Bool tensors
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
        );
    }
    
    auto float_fn = [](float x) { return -x; };
    auto double_fn = [](double x) { return -x; };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// ============================================================================
// ABSOLUTE - Now throws error for Bool tensors (matching PyTorch)
// ============================================================================

Tensor absolute_out_cpu_wrap(const Tensor& input_tensor) {
    // PyTorch behavior: abs() not implemented for Bool tensors
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"abs_cpu\" not implemented for 'Bool'"
        );
    }
    
    auto float_fn = [](float x) { return fabsf(x); };
    auto double_fn = [](double x) { return std::fabs(x); };
    return generic_unary_out_cpu(input_tensor, input_tensor.dtype(), float_fn, double_fn);
}

void absolute_in_cpu_wrap(Tensor& input_tensor) {
    // PyTorch behavior: abs() not implemented for Bool tensors
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"abs_cpu\" not implemented for 'Bool'"
        );
    }
    
    auto float_fn = [](float x) { return fabsf(x); };
    auto double_fn = [](double x) { return std::fabs(x); };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// ============================================================================
// SIGN
// ============================================================================

Tensor sign_out_cpu_wrap(const Tensor& input_tensor) {
    auto float_fn = [](float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); };
    auto double_fn = [](double x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); };
    return generic_unary_out_cpu(input_tensor, input_tensor.dtype(), float_fn, double_fn);
}

void sign_in_cpu_wrap(Tensor& input_tensor) {
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"sign\" not implemented for 'Bool'"
        );
    }
    auto float_fn = [](float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); };
    auto double_fn = [](double x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// ============================================================================
// POWER
// ============================================================================

// Integer exponent version
Tensor power_out_cpu_wrap(const Tensor& input_tensor, int exponent) {
    auto float_fn = [exponent](float x) { 
        return safe_pow(x, static_cast<float>(exponent)); 
    };
    auto double_fn = [exponent](double x) { 
        return safe_pow(x, static_cast<double>(exponent)); 
    };
    return generic_unary_out_cpu(input_tensor, get_promoted_dtype(input_tensor.dtype()), 
                                  float_fn, double_fn);
}

void power_in_cpu_wrap(Tensor& input_tensor, int exponent) {
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"pow\" not implemented for 'Bool'"
        );
    }
    
    if(exponent < 0) {
        throw std::runtime_error(
            "Inplace power operations with negative exponents are not supported. "
            "Use out-of-place power operation instead."
        );
    }
    auto float_fn = [exponent](float x) { 
        return safe_pow(x, static_cast<float>(exponent)); 
    };
    auto double_fn = [exponent](double x) { 
        return safe_pow(x, static_cast<double>(exponent)); 
    };
    generic_unary_in_cpu(input_tensor, float_fn, double_fn);
}

// Float exponent version
Tensor power_out_cpu_wrap(const Tensor& input_tensor, float exponent) {
    auto float_fn = [exponent](float x) { 
        return safe_pow(x, exponent); 
    };
    auto double_fn = [exponent](double x) { 
        return safe_pow(x, static_cast<double>(exponent)); 
    };
    return generic_unary_out_cpu(input_tensor, get_promoted_dtype(input_tensor.dtype()), 
                                  float_fn, double_fn);
}

void power_in_cpu_wrap([[maybe_unused]] Tensor& input_tensor,[[maybe_unused]] float exponent) {
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"pow\" not implemented for 'Bool'"
        );
    }
    throw std::runtime_error(
            "Inplace power operations is accepted only for int exponent values. "
            "Use out-of-place power operation instead."
        );
}

// Double exponent version
Tensor power_out_cpu_wrap(const Tensor& input_tensor, double exponent) {
    auto float_fn = [exponent](float x) { 
        return safe_pow(x, static_cast<float>(exponent)); 
    };
    auto double_fn = [exponent](double x) { 
        return safe_pow(x, exponent); 
    };
    return generic_unary_out_cpu(input_tensor, get_promoted_dtype(input_tensor.dtype()), 
                                  float_fn, double_fn);
}

void power_in_cpu_wrap([[maybe_unused]]Tensor& input_tensor, [[maybe_unused]]double exponent) {
    if (input_tensor.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"pow\" not implemented for 'Bool'"
        );
    }
    throw std::runtime_error(
            "Inplace power operations is accepted only for int exponent values. "
            "Use out-of-place power operation instead."
        );
}

} // namespace OwnTensor