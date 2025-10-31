#include <cmath>
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/exp_log.hpp"
#include "dtype/DtypeCastUtils.h"
#include "dtype/DtypeTraits.h"  // ← ADD THIS for is_int()

namespace OwnTensor {
// ============================================================================
// Function Pointers for CPU Math Operations
// ============================================================================
static inline float expf_fn(float x) { return expf(x); }
static inline double exp_fn(double x) { return std::exp(x); }
static inline float exp2f_fn(float x) { return exp2f(x); }
static inline double exp2_fn(double x) { return std::exp2(x); }
static inline float logf_fn(float x) { return logf(x); }
static inline double log_fn(double x) { return std::log(x); }
static inline float log2f_fn(float x) { return log2f(x); }
static inline double log2_fn(double x) { return std::log2(x); }
static inline float log10f_fn(float x) { return log10f(x); }
static inline double log10_fn(double x) { return std::log10(x); }
// ============================================================================
// Generic Unary Kernel for CPU
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
// Generic Out-of-Place CPU Wrapper Using dispatch_by_dtype - FIXED VERSION
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
Tensor generic_unary_out_cpu(const Tensor& input_tensor) {
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
    // IMPORTANT: dispatch_by_dtype passes the ACTUAL TYPE instance
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
// Generic In-Place CPU Wrapper Using dispatch_by_dtype - FIXED VERSION
// ============================================================================
template<float(*FloatFunc)(float), double(*DoubleFunc)(double)>
void generic_unary_in_cpu(Tensor& input_tensor) {
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
    // IMPORTANT: dispatch_by_dtype passes the ACTUAL TYPE, not a type_tag!
    dispatch_by_dtype(input_tensor.dtype(), [&](auto type_instance) {
        // type_instance is already float16_t/bfloat16_t/float/double/int16_t/etc.
        // NOT a type tag! So we use decltype to get its type
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
// CPU Wrapper Functions - REMOVE TRY-CATCH BLOCKS
// ============================================================================
Tensor exp_out_cpu_wrap(const Tensor& input) {
    return generic_unary_out_cpu<expf_fn, exp_fn>(input);
}

void exp_in_cpu_wrap(Tensor& input) {
    generic_unary_in_cpu<expf_fn, exp_fn>(input);
}

Tensor exp2_out_cpu_wrap(const Tensor& input) {
    return generic_unary_out_cpu<exp2f_fn, exp2_fn>(input);
}

void exp2_in_cpu_wrap(Tensor& input) {
    generic_unary_in_cpu<exp2f_fn, exp2_fn>(input);
}

Tensor log_out_cpu_wrap(const Tensor& input) {
    return generic_unary_out_cpu<logf_fn, log_fn>(input);
}

void log_in_cpu_wrap(Tensor& input) {
    generic_unary_in_cpu<logf_fn, log_fn>(input);
}

Tensor log2_out_cpu_wrap(const Tensor& input) {
    return generic_unary_out_cpu<log2f_fn, log2_fn>(input);
}

void log2_in_cpu_wrap(Tensor& input) {
    generic_unary_in_cpu<log2f_fn, log2_fn>(input);
}

Tensor log10_out_cpu_wrap(const Tensor& input) {
    return generic_unary_out_cpu<log10f_fn, log10_fn>(input);
}

void log10_in_cpu_wrap(Tensor& input) {
    generic_unary_in_cpu<log10f_fn, log10_fn>(input);
}

} // namespace OwnTensor