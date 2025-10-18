#include <Tensor.h>
#include <iostream>


namespace OwnTensor{
// promotion level logic

inline int get_dtype_order(Dtype dtype) {
    switch(dtype) {
        case Dtype::Int16:   return 0;
        case Dtype::Int32:   return 1;
        case Dtype::Int64:   return 2;
        case Dtype::Bfloat16:return 3;
        case Dtype::Float16: return 4;
        case Dtype::Float32: return 5;
        case Dtype::Float64: return 6;
        default:             return -1;
    }
}
inline Dtype scalar_to_dtype(const std::type_info& tinfo) {
    if (tinfo == typeid(int16_t)) return Dtype::Int16;
    if (tinfo == typeid(int32_t)) return Dtype::Int32;
    if (tinfo == typeid(int64_t)) return Dtype::Int64;
    if (tinfo == typeid(int)) return Dtype::Int32;      // int -> int32
    if (tinfo == typeid(float)) return Dtype::Float32;
    if (tinfo == typeid(double)) return Dtype::Float64; // float -> float64 by default
    
    throw std::runtime_error("Unsupported scalar type in scalar_to_dtype");
}

inline Dtype promote_dtype(Dtype a, Dtype b) {
    if (a == b) return a;

    // If both are integer, promote to wider
    if (!is_float(a) && !is_float(b))
        return (get_dtype_order(a) > get_dtype_order(b)) ? a : b;

    // If both are floats, handle bf16-f16 specially
    if (is_float(a) && is_float(b)) {
        if ((a == Dtype::Bfloat16 && b == Dtype::Float16) ||
            (a == Dtype::Float16 && b == Dtype::Bfloat16))
            return Dtype::Float32;
        return (get_dtype_order(a) > get_dtype_order(b)) ? a : b;
    }

    // If mixed int/float, always promote to wider float
    Dtype float_type = is_float(a) ? a : b;
    Dtype int_type = is_float(a) ? b : a;

    // Handle bf16-f16 mixed with int: treat as wider float
    if ((float_type == Dtype::Bfloat16 && int_type == Dtype::Int16) ||
        (float_type == Dtype::Float16 && int_type == Dtype::Int16)) {
        return float_type; // Typically bf16/f16 + int -> bf16/f16
    }
    if (float_type == Dtype::Bfloat16 && int_type == Dtype::Float16)
        return Dtype::Float32;

    return float_type;
}

    // Dispatcher helper
template<typename TensorT, typename ScalarT, typename OutT, typename Op>
void dispatch_operation(const void* in_data, void* out_data, size_t total, ScalarT scalar, Op op) {
    const TensorT* in_ptr = static_cast<const TensorT*>(in_data);
    OutT* out_ptr = static_cast<OutT*>(out_data);
    for (size_t i = 0; i < total; ++i) {
        out_ptr[i] = op(static_cast<OutT>(in_ptr[i]), static_cast<OutT>(scalar));
    }
}

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
Tensor operator+(const Tensor& tensor, T scalar) {
    Dtype scalar_dtype = scalar_to_dtype(typeid(T));
    Dtype out_dtype = promote_dtype(tensor.dtype(), scalar_dtype);
    
    Tensor output(tensor.shape(), out_dtype, tensor.device(), tensor.requires_grad());
    size_t total = tensor.numel();
    
    // Dispatch based on tensor dtype and output dtype
    #define DISPATCH_ADD(TENSOR_TYPE, OUT_TYPE, tensor_t, out_t) \
        if (tensor.dtype() == TENSOR_TYPE && out_dtype == OUT_TYPE) { \
            dispatch_operation<tensor_t, T, out_t>(tensor.data(), output.data(), total, scalar, \
                [](auto a, auto b) { return a + b; }); \
            return output; \
        }
    
    // All combinations (tensor dtype -> output dtype)
    DISPATCH_ADD(Dtype::Int16, Dtype::Int16, int16_t, int16_t)
    DISPATCH_ADD(Dtype::Int16, Dtype::Int32, int16_t, int32_t)
    DISPATCH_ADD(Dtype::Int16, Dtype::Int64, int16_t, int64_t)
    DISPATCH_ADD(Dtype::Int16, Dtype::Float32, int16_t, float)
    DISPATCH_ADD(Dtype::Int16, Dtype::Float64, int16_t, double)
    
    DISPATCH_ADD(Dtype::Int32, Dtype::Int32, int32_t, int32_t)
    DISPATCH_ADD(Dtype::Int32, Dtype::Int64, int32_t, int64_t)
    DISPATCH_ADD(Dtype::Int32, Dtype::Float32, int32_t, float)
    DISPATCH_ADD(Dtype::Int32, Dtype::Float64, int32_t, double)
    
    DISPATCH_ADD(Dtype::Int64, Dtype::Int64, int64_t, int64_t)
    DISPATCH_ADD(Dtype::Int64, Dtype::Float32, int64_t, float)
    DISPATCH_ADD(Dtype::Int64, Dtype::Float64, int64_t, double)
    
    DISPATCH_ADD(Dtype::Float16, Dtype::Float16, float16_t, float16_t)
    DISPATCH_ADD(Dtype::Float16, Dtype::Float32, float16_t, float)
    DISPATCH_ADD(Dtype::Float16, Dtype::Float64, float16_t, double)
    
    DISPATCH_ADD(Dtype::Bfloat16, Dtype::Bfloat16, bfloat16_t, bfloat16_t)
    DISPATCH_ADD(Dtype::Bfloat16, Dtype::Float32, bfloat16_t, float)
    DISPATCH_ADD(Dtype::Bfloat16, Dtype::Float64, bfloat16_t, double)
    
    DISPATCH_ADD(Dtype::Float32, Dtype::Float32, float, float)
    DISPATCH_ADD(Dtype::Float32, Dtype::Float64, float, double)
    
    DISPATCH_ADD(Dtype::Float64, Dtype::Float64, double, double)
    
    #undef DISPATCH_ADD
    
    throw std::runtime_error("Unsupported dtype combination for operator+");
}
// Allow scalar + tensor
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
Tensor operator+(T scalar, const Tensor& tensor) {
    return tensor + scalar;
}
// Remove these duplicates - keep only one per actual type
template Tensor operator+(const Tensor&, int16_t);
template Tensor operator+(int16_t, const Tensor&);

// Keep EITHER int OR int32_t, not both (they're the same on your system)
template Tensor operator+(const Tensor&, int);
template Tensor operator+(int, const Tensor&);

template Tensor operator+(const Tensor&, int64_t);
template Tensor operator+(int64_t, const Tensor&);

template Tensor operator+(const Tensor&, float);
template Tensor operator+(float, const Tensor&);

template Tensor operator+(const Tensor&, double);
template Tensor operator+(double, const Tensor&);

}