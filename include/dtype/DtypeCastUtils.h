#pragma once

#include "core/Tensor.h"  // ensures Dtype, Tensor, shape/device APIs are visible

namespace OwnTensor {

// Integer promotion policy
inline constexpr Dtype get_promoted_dtype(Dtype input_dtype) {
    switch (input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Bool:
            return Dtype::Float32;
        case Dtype::Int64:
            return Dtype::Float64;
        default:
            return input_dtype; // no promotion for float types
    }
}

// Convert bf16/f16 tensor to Float32 (CPU path)
inline Tensor convert_half_to_float32(const Tensor& input) {
    Tensor temp(input.shape(), Dtype::Float32, input.device(), input.requires_grad()); // [file:3]
    float* temp_ptr = temp.data<float>(); // [file:3]

    if (input.dtype() == Dtype::Float16) {
        const float16_t* in_ptr = input.data<float16_t>(); // [file:3]
        for (size_t i = 0; i < input.numel(); ++i) { // [file:3]
            temp_ptr[i] = static_cast<float>(in_ptr[i]);
        }
    } else { // Bfloat16
        const bfloat16_t* in_ptr = input.data<bfloat16_t>(); // [file:3]
        for (size_t i = 0; i < input.numel(); ++i) { // [file:3]
            temp_ptr[i] = static_cast<float>(in_ptr[i]);
        }
    }
    return temp;
}

// Convert Float32 tensor back to bf16/f16 (CPU path)
inline void convert_float32_to_half(const Tensor& float_tensor, Tensor& output) {
    const float* float_ptr = float_tensor.data<float>(); // [file:3]

    if (output.dtype() == Dtype::Float16) {
        float16_t* out_ptr = output.data<float16_t>(); // [file:3]
        for (size_t i = 0; i < output.numel(); ++i) { // [file:3]
            out_ptr[i] = float16_t(float_ptr[i]);
        }
    } else { // Bfloat16
        bfloat16_t* out_ptr = output.data<bfloat16_t>(); // [file:3]
        for (size_t i = 0; i < output.numel(); ++i) { // [file:3]
            out_ptr[i] = bfloat16_t(float_ptr[i]);
        }
    }
}

// Promote to Float64 for square operation
// inline constexpr Dtype get_promoted_dtype_float64(Dtype input_dtype) {
//     switch (input_dtype) {
//         case Dtype::Int16:
//         case Dtype::Int32:
//         case Dtype::Int64:
//             return Dtype::Float64;
//         default:
//             return input_dtype; // no promotion for float types
//     }
// }

// promote dtype for square (Int -> Float64)
inline Dtype get_promoted_dtype_square(Dtype input_dtype) {
    switch(input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
        case Dtype::Bool:
            return Dtype::Float64;
        default:
            return input_dtype;
    }
}

// Helper function to handle edge cases
template<typename T, typename ExpT>
inline T safe_pow(T base, ExpT exponent) {
    // Handle special cases
    if (std::isnan(base) || std::isnan(exponent)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    
    // 0^0 returns 1 (convention)
    if (base == T(0) && exponent == ExpT(0)) {
        return T(1);
    }
    
    // 0^(negative) returns infinity
    if (base == T(0) && exponent < ExpT(0)) {
        return std::numeric_limits<T>::infinity();
    }
    
    // 0^(positive) returns 0
    if (base == T(0) && exponent > ExpT(0)) {
        return T(0);
    }
    
    // Negative base with non-integer exponent returns NaN
    if (base < T(0) && std::floor(exponent) != exponent) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    
    // Standard power computation
    T result = std::pow(base, static_cast<T>(exponent));
    
    // Check for overflow/underflow
    if (std::isinf(result) || result == T(0)) {
        return result; // Let inf/0 propagate
    }
    
    return result;
}
} // namespace OwnTensor