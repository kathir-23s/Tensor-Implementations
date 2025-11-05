#include <cmath>
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/Trigonometry.hpp"
#include "dtype/DtypeCastUtils.h"
#include "dtype/DtypeTraits.h"

namespace OwnTensor {
// ============================================================================
// Function Pointers for CPU Trigonometric Operations
// ============================================================================
static inline float sinf_fn(float x) { return std::sin(x); }
static inline double sin_fn(double x) { return std::sin(x); }
static inline float cosf_fn(float x) { return std::cos(x); }
static inline double cos_fn(double x) { return std::cos(x); }
static inline float tanf_fn(float x) { return std::tan(x); }
static inline double tan_fn(double x) { return std::tan(x); }

static inline float asinf_fn(float x) { return std::asin(x); }
static inline double asin_fn(double x) { return std::asin(x); }
static inline float acosf_fn(float x) { return std::acos(x); }
static inline double acos_fn(double x) { return std::acos(x); }
static inline float atanf_fn(float x) { return std::atan(x); }
static inline double atan_fn(double x) { return std::atan(x); }

static inline float sinhf_fn(float x) { return std::sinh(x); }
static inline double sinh_fn(double x) { return std::sinh(x); }
static inline float coshf_fn(float x) { return std::cosh(x); }
static inline double cosh_fn(double x) { return std::cosh(x); }
static inline float tanhf_fn(float x) { return std::tanh(x); }
static inline double tanh_fn(double x) { return std::tanh(x); }

static inline float asinhf_fn(float x) { return std::asinh(x); }
static inline double asinh_fn(double x) { return std::asinh(x); }
static inline float acoshf_fn(float x) { return std::acosh(x); }
static inline double acosh_fn(double x) { return std::acosh(x); }
static inline float atanhf_fn(float x) { return std::atanh(x); }
static inline double atanh_fn(double x) { return std::atanh(x); }

// ============================================================================
// Generic Unary Kernel for CPU - REUSE FROM EXPONENTS
// ============================================================================
template<typename T_In, typename T_Out, T_Out(*Func)(T_Out)>
void unary_kernel_cpu(const T_In* in, T_Out* out, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        T_Out temp_val = static_cast<T_Out>(in[i]);
        out[i] = Func(temp_val);
    }
}

// ============================================================================
// Generic Out-of-Place CPU Wrapper for Trigonometric Functions
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
Tensor generic_trigonometric_out_cpu(const Tensor& input_tensor) {
    // Handle bf16/f16 by promoting to float32
    if (input_tensor.dtype() == Dtype::Bfloat16 || input_tensor.dtype() == Dtype::Float16) {
        Dtype original_dtype = input_tensor.dtype();
        Tensor temp_input = convert_half_to_float32(input_tensor);
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        
        unary_kernel_cpu<float, float, FloatFunc>(
            temp_input.data<float>(),
            temp_output.data<float>(),
            input_tensor.numel()
        );
        
        Tensor output(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        convert_float32_to_half(temp_output, output);
        return output;
    }
    
    // Determine output dtype (promote integers)
    Dtype output_dtype = get_promoted_dtype(input_tensor.dtype());
    Tensor output(input_tensor.shape(), output_dtype, input_tensor.device(), input_tensor.requires_grad());
    
    // Use dispatch_by_dtype to handle input dtype
    dispatch_by_dtype(input_tensor.dtype(), [&](auto in_type_instance) {
        using InputType = decltype(in_type_instance);
        
        // Use dispatch_by_dtype to handle output dtype
        dispatch_by_dtype(output_dtype, [&](auto out_type_instance) {
            using OutputType = decltype(out_type_instance);
            
            // Select appropriate function based on output type
            if constexpr (std::is_same_v<OutputType, float>) {
                unary_kernel_cpu<InputType, OutputType, FloatFunc>(
                    input_tensor.data<InputType>(),
                    output.data<OutputType>(),
                    input_tensor.numel()
                );
            } else if constexpr (std::is_same_v<OutputType, double>) {
                unary_kernel_cpu<InputType, OutputType, DoubleFunc>(
                    input_tensor.data<InputType>(),
                    output.data<OutputType>(),
                    input_tensor.numel()
                );
            }
        });
    });
    return output;
}

// ============================================================================
// Generic In-Place CPU Wrapper for Trigonometric Functions
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
void generic_trigonometric_in_cpu(Tensor& input_tensor) {
    // Handle bf16/f16 by promoting to float32
    if (input_tensor.dtype() == Dtype::Bfloat16 || input_tensor.dtype() == Dtype::Float16) {
        [[maybe_unused]] Dtype original_dtype = input_tensor.dtype();
        Tensor temp_input = convert_half_to_float32(input_tensor);
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        
        unary_kernel_cpu<float, float, FloatFunc>(
            temp_input.data<float>(),
            temp_output.data<float>(),
            input_tensor.numel()
        );
        
        convert_float32_to_half(temp_output, input_tensor);
        return;
    }
    // Reject integer types for in-place operations
    if (is_int(input_tensor.dtype())) {
        throw std::runtime_error("Error: cannot do inplace operations for integer data types!");
    }
    // Use dispatch_by_dtype for float types
    dispatch_by_dtype(input_tensor.dtype(), [&](auto type_instance) {
        using DataType = decltype(type_instance);
        
        if constexpr (std::is_same_v<DataType, float>) {
            unary_kernel_cpu<DataType, DataType, FloatFunc>(
                input_tensor.data<DataType>(),
                input_tensor.data<DataType>(),
                input_tensor.numel()
            );
        } else if constexpr (std::is_same_v<DataType, double>) {
            unary_kernel_cpu<DataType, DataType, DoubleFunc>(
                input_tensor.data<DataType>(),
                input_tensor.data<DataType>(),
                input_tensor.numel()
            );
        }
    });
}

// ============================================================================
// CPU Wrapper Functions - Trigonometric Operations
// ============================================================================

// Basic trigonometric functions
Tensor sin_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<sinf_fn, sin_fn>(input);
}

void sin_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<sinf_fn, sin_fn>(input);
}

Tensor cos_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<cosf_fn, cos_fn>(input);
}

void cos_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<cosf_fn, cos_fn>(input);
}

Tensor tan_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<tanf_fn, tan_fn>(input);
}

void tan_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<tanf_fn, tan_fn>(input);
}

// Inverse trigonometric functions
Tensor asin_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<asinf_fn, asin_fn>(input);
}

void asin_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<asinf_fn, asin_fn>(input);
}

Tensor acos_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<acosf_fn, acos_fn>(input);
}

void acos_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<acosf_fn, acos_fn>(input);
}

Tensor atan_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<atanf_fn, atan_fn>(input);
}

void atan_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<atanf_fn, atan_fn>(input);
}

// Hyperbolic functions
Tensor sinh_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<sinhf_fn, sinh_fn>(input);
}

void sinh_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<sinhf_fn, sinh_fn>(input);
}

Tensor cosh_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<coshf_fn, cosh_fn>(input);
}

void cosh_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<coshf_fn, cosh_fn>(input);
}

Tensor tanh_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<tanhf_fn, tanh_fn>(input);
}

void tanh_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<tanhf_fn, tanh_fn>(input);
}

// Inverse hyperbolic functions
Tensor asinh_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<asinhf_fn, asinh_fn>(input);
}

void asinh_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<asinhf_fn, asinh_fn>(input);
}

Tensor acosh_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<acoshf_fn, acosh_fn>(input);
}

void acosh_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<acoshf_fn, acosh_fn>(input);
}

Tensor atanh_out_cpu_wrap(const Tensor& input) {
    return generic_trigonometric_out_cpu<atanhf_fn, atanh_fn>(input);
}

void atanh_in_cpu_wrap(Tensor& input) {
    generic_trigonometric_in_cpu<atanhf_fn, atanh_fn>(input);
}

} // namespace OwnTensor