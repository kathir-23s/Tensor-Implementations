// ScalarOps.cpp — CPU path (no macros)
#include "ops/ScalarOps.h"
#include "core/TensorDispatch.h"   // dispatch_by_dtype + Dtype
#include "dtype/Types.h"            // Dtype enum if not already included by Tensor.h
#include <type_traits>
#include <stdexcept>
#include <cstring>            // memcpy for safety

namespace OwnTensor {

// ---- use your existing precise conversions from the uploaded file ----
inline float load_as_float(uint16_t bits, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float16_to_float(bits);
    if (dt == Dtype::Bfloat16) return detail::bfloat16_to_float(bits);
    return static_cast<float>(bits);
}
inline uint16_t store_from_float(float v, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float_to_float16(v);
    if (dt == Dtype::Bfloat16) return detail::float_to_bfloat16(v);
    return static_cast<uint16_t>(v);
}

template <typename T>
inline float load_as_float_typed(const T* p, size_t i, Dtype) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        // unreachable for non-fp16/bf16 if your traits map only those to u16
        return static_cast<float>(p[i]);
    } else {
        return static_cast<float>(p[i]);
    }
}
template <>
inline float load_as_float_typed<uint16_t>(const uint16_t* p, size_t i, Dtype dt) {
    return load_as_float(p[i], dt);
}

template <typename T>
inline void store_from_float_typed(T* p, size_t i, float v, Dtype) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        p[i] = static_cast<uint16_t>(v);
    } else {
        p[i] = static_cast<T>(v);
    }
}
template <>
inline void store_from_float_typed<uint16_t>(uint16_t* p, size_t i, float v, Dtype dt) {
    p[i] = store_from_float(v, dt);
}

// ------------------------- In-place operators -------------------------

template <typename S>
Tensor& operator+=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(data, i, dt);
            store_from_float_typed<T>(data, i, v + s, dt);
        }
    });
    return tensor;
}

template <typename S>
Tensor& operator-=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(data, i, dt);
            store_from_float_typed<T>(data, i, v - s, dt);
        }
    });
    return tensor;
}

template <typename S>
Tensor& operator*=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(data, i, dt);
            store_from_float_typed<T>(data, i, v * s, dt);
        }
    });
    return tensor;
}

template <typename S>
Tensor& operator/=(Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T>) {
            if (static_cast<double>(scalar) == 0.0)
                throw std::runtime_error("Division by zero");
        }
        T* data = tensor.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(data, i, dt);
            store_from_float_typed<T>(data, i, v / s, dt);
        }
    });
    return tensor;
}

// --------------------- Tensor (lhs) ⊗ Scalar (rhs) --------------------

template <typename S>
Tensor operator+(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(src, i, dt);
            store_from_float_typed<T>(dst, i, v + s, dt);
        }
    });
    return out;
}

template <typename S>
Tensor operator-(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(src, i, dt);
            store_from_float_typed<T>(dst, i, v - s, dt);
        }
    });
    return out;
}

template <typename S>
Tensor operator*(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(src, i, dt);
            store_from_float_typed<T>(dst, i, v * s, dt);
        }
    });
    return out;
}

template <typename S>
Tensor operator/(const Tensor& tensor, S scalar) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("GPU operations not implemented in CPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        if constexpr (std::is_integral_v<T>) {
            if (static_cast<double>(scalar) == 0.0)
                throw std::runtime_error("Division by zero");
        }
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(src, i, dt);
            store_from_float_typed<T>(dst, i, v / s, dt);
        }
    });
    return out;
}

// --------------------- Scalar (lhs) ⊗ Tensor (rhs) --------------------

template <typename S>
Tensor operator+(S scalar, const Tensor& tensor) { return tensor + scalar; }

template <typename S>
Tensor operator-(S scalar, const Tensor& tensor) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(src, i, dt);
            store_from_float_typed<T>(dst, i, s - v, dt);
        }
    });
    return out;
}

template <typename S>
Tensor operator*(S scalar, const Tensor& tensor) { return tensor * scalar; }

template <typename S>
Tensor operator/(S scalar, const Tensor& tensor) {
    if (!tensor.device().is_cpu())
        throw std::runtime_error("CPU operations not implemented in GPU file");

    Tensor out(tensor.shape(), tensor.dtype(), tensor.device(), tensor.requires_grad());
    dispatch_by_dtype(tensor.dtype(), [&](auto dummy){
        using T = decltype(dummy);
        const T* src = tensor.data<T>();
        T* dst = out.data<T>();
        const size_t n = tensor.numel();
        const Dtype dt = tensor.dtype();

        if constexpr (std::is_integral_v<T>) {
            for (size_t i = 0; i < n; ++i) {
                if (src[i] == static_cast<T>(0))
                    throw std::runtime_error("Division by zero");
            }
        }
        const float s = static_cast<float>(scalar);
        for (size_t i = 0; i < n; ++i) {
            const float v = load_as_float_typed<T>(src, i, dt);
            store_from_float_typed<T>(dst, i, s / v, dt);
        }
    });
    return out;
}

// -------------------- explicit instantiations (no macros) --------------------

template Tensor& operator+=<int16_t>(Tensor&, int16_t);
template Tensor& operator+=<int32_t>(Tensor&, int32_t);
template Tensor& operator+=<int64_t>(Tensor&, int64_t);
template Tensor& operator+=<float>(Tensor&, float);
template Tensor& operator+=<double>(Tensor&, double);
template Tensor& operator+=<float16_t>(Tensor&, float16_t);
template Tensor& operator+=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator-=<int16_t>(Tensor&, int16_t);
template Tensor& operator-=<int32_t>(Tensor&, int32_t);
template Tensor& operator-=<int64_t>(Tensor&, int64_t);
template Tensor& operator-=<float>(Tensor&, float);
template Tensor& operator-=<double>(Tensor&, double);
template Tensor& operator-=<float16_t>(Tensor&, float16_t);
template Tensor& operator-=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator*=<int16_t>(Tensor&, int16_t);
template Tensor& operator*=<int32_t>(Tensor&, int32_t);
template Tensor& operator*=<int64_t>(Tensor&, int64_t);
template Tensor& operator*=<float>(Tensor&, float);
template Tensor& operator*=<double>(Tensor&, double);
template Tensor& operator*=<float16_t>(Tensor&, float16_t);
template Tensor& operator*=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator/=<int16_t>(Tensor&, int16_t);
template Tensor& operator/=<int32_t>(Tensor&, int32_t);
template Tensor& operator/=<int64_t>(Tensor&, int64_t);
template Tensor& operator/=<float>(Tensor&, float);
template Tensor& operator/=<double>(Tensor&, double);
template Tensor& operator/=<float16_t>(Tensor&, float16_t);
template Tensor& operator/=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor operator+<int16_t>(const Tensor&, int16_t);
template Tensor operator+<int32_t>(const Tensor&, int32_t);
template Tensor operator+<int64_t>(const Tensor&, int64_t);
template Tensor operator+<float>(const Tensor&, float);
template Tensor operator+<double>(const Tensor&, double);
template Tensor operator+<float16_t>(const Tensor&, float16_t);
template Tensor operator+<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator-<int16_t>(const Tensor&, int16_t);
template Tensor operator-<int32_t>(const Tensor&, int32_t);
template Tensor operator-<int64_t>(const Tensor&, int64_t);
template Tensor operator-<float>(const Tensor&, float);
template Tensor operator-<double>(const Tensor&, double);
template Tensor operator-<float16_t>(const Tensor&, float16_t);
template Tensor operator-<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator*<int16_t>(const Tensor&, int16_t);
template Tensor operator*<int32_t>(const Tensor&, int32_t);
template Tensor operator*<int64_t>(const Tensor&, int64_t);
template Tensor operator*<float>(const Tensor&, float);
template Tensor operator*<double>(const Tensor&, double);
template Tensor operator*<float16_t>(const Tensor&, float16_t);
template Tensor operator*<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator/<int16_t>(const Tensor&, int16_t);
template Tensor operator/<int32_t>(const Tensor&, int32_t);
template Tensor operator/<int64_t>(const Tensor&, int64_t);
template Tensor operator/<float>(const Tensor&, float);
template Tensor operator/<double>(const Tensor&, double);
template Tensor operator/<float16_t>(const Tensor&, float16_t);
template Tensor operator/<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator+<int16_t>(int16_t, const Tensor&);
template Tensor operator+<int32_t>(int32_t, const Tensor&);
template Tensor operator+<int64_t>(int64_t, const Tensor&);
template Tensor operator+<float>(float, const Tensor&);
template Tensor operator+<double>(double, const Tensor&);
template Tensor operator+<float16_t>(float16_t, const Tensor&);
template Tensor operator+<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator-<int16_t>(int16_t, const Tensor&);
template Tensor operator-<int32_t>(int32_t, const Tensor&);
template Tensor operator-<int64_t>(int64_t, const Tensor&);
template Tensor operator-<float>(float, const Tensor&);
template Tensor operator-<double>(double, const Tensor&);
template Tensor operator-<float16_t>(float16_t, const Tensor&);
template Tensor operator-<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator*<int16_t>(int16_t, const Tensor&);
template Tensor operator*<int32_t>(int32_t, const Tensor&);
template Tensor operator*<int64_t>(int64_t, const Tensor&);
template Tensor operator*<float>(float, const Tensor&);
template Tensor operator*<double>(double, const Tensor&);
template Tensor operator*<float16_t>(float16_t, const Tensor&);
template Tensor operator*<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator/<int16_t>(int16_t, const Tensor&);
template Tensor operator/<int32_t>(int32_t, const Tensor&);
template Tensor operator/<int64_t>(int64_t, const Tensor&);
template Tensor operator/<float>(float, const Tensor&);
template Tensor operator/<double>(double, const Tensor&);
template Tensor operator/<float16_t>(float16_t, const Tensor&);
template Tensor operator/<bfloat16_t>(bfloat16_t, const Tensor&);

} // namespace OwnTensor

