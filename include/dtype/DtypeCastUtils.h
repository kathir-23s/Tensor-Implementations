#pragma once

#include "core/Tensor.h"  // ensures Dtype, Tensor, shape/device APIs are visible

namespace OwnTensor {

// Integer promotion policy
inline constexpr Dtype get_promoted_dtype(Dtype input_dtype) {
    switch (input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
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
        for (int i = 0; i < input.numel(); ++i) { // [file:3]
            temp_ptr[i] = static_cast<float>(in_ptr[i]);
        }
    } else { // Bfloat16
        const bfloat16_t* in_ptr = input.data<bfloat16_t>(); // [file:3]
        for (int i = 0; i < input.numel(); ++i) { // [file:3]
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
        for (int i = 0; i < output.numel(); ++i) { // [file:3]
            out_ptr[i] = float16_t(float_ptr[i]);
        }
    } else { // Bfloat16
        bfloat16_t* out_ptr = output.data<bfloat16_t>(); // [file:3]
        for (int i = 0; i < output.numel(); ++i) { // [file:3]
            out_ptr[i] = bfloat16_t(float_ptr[i]);
        }
    }
}

// Promote to Float64 for square operation
inline constexpr Dtype get_promoted_dtype_float64(Dtype input_dtype) {
    switch (input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
            return Dtype::Float64;
        default:
            return input_dtype; // no promotion for float types
    }
}

// promote dtype for square (Int -> Float64)
inline Dtype get_promoted_dtype_square(Dtype input_dtype) {
    switch(input_dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
            return Dtype::Float64;
        default:
            return input_dtype;
    }
}

} // namespace OwnTensor