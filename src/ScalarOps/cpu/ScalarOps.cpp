// ScalarOps.cpp (CPU backend)
#include <cstdint>
#include <stdexcept>
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"

namespace OwnTensor {
namespace { // file-local helpers

inline bool is_integer_dtype(Dtype dt) {
    return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
}

// conversions supplied by your numeric utilities
namespace detail {
    float    float16_to_float(uint16_t);
    uint16_t float_to_float16(float);
    float    bfloat16_to_float(uint16_t);
    uint16_t float_to_bfloat16(float);
}

// uint16_t-backed half/bfloat16 helpers
inline float load_u16_as_f32(uint16_t bits, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float16_to_float(bits);
    if (dt == Dtype::Bfloat16) return detail::bfloat16_to_float(bits);
    return static_cast<float>(bits);
}
inline uint16_t store_f32_to_u16(float v, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float_to_float16(v);
    if (dt == Dtype::Bfloat16) return detail::float_to_bfloat16(v);
    return static_cast<uint16_t>(v);
}

template <typename T>
inline double ld(const T* p, size_t i, Dtype) { return static_cast<double>(p[i]); }
template <>
inline double ld<uint16_t>(const uint16_t* p, size_t i, Dtype dt) {
    return static_cast<double>(load_u16_as_f32(p[i], dt));
}

template <typename T>
inline void st(T* p, size_t i, double v, Dtype) { p[i] = static_cast<T>(v); }
template <>
inline void st<uint16_t>(uint16_t* p, size_t i, double v, Dtype dt) {
    p[i] = store_f32_to_u16(static_cast<float>(v), dt);
}

template <typename T, typename F>
inline void apply_inplace(T* data, size_t n, Dtype dt, F&& f) {
    for (size_t i = 0; i < n; ++i) st<T>(data, i, f(ld<T>(data, i, dt)), dt);
}
template <typename T, typename F>
inline void apply_copy(const T* src, T* dst, size_t n, Dtype dt, F&& f) {
    for (size_t i = 0; i < n; ++i) st<T>(dst, i, f(ld<T>(src, i, dt)), dt);
}

} // anon

// --------- public CPU backend (Int16/32/64 + F16/BF16/F32/F64) ---------
void cpu_add_inplace(Tensor& t, double s) {
    const Dtype dt = t.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_inplace<T>(t.data<T>(), t.numel(), dt, [=](double v){ return v + s; });
    });
}
void cpu_sub_inplace(Tensor& t, double s) {
    const Dtype dt = t.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_inplace<T>(t.data<T>(), t.numel(), dt, [=](double v){ return v - s; });
    });
}
void cpu_mul_inplace(Tensor& t, double s) {
    const Dtype dt = t.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_inplace<T>(t.data<T>(), t.numel(), dt, [=](double v){ return v * s; });
    });
}
void cpu_div_inplace(Tensor& t, double s) {
    const Dtype dt = t.dtype();
    if (is_integer_dtype(dt) && s == 0.0) throw std::runtime_error("Division by zero");
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_inplace<T>(t.data<T>(), t.numel(), dt, [=](double v){ return v / s; });
    });
}

Tensor cpu_add_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), dt, [=](double v){ return v + s; });
    });
    return out;
}
Tensor cpu_sub_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), dt, [=](double v){ return v - s; });
    });
    return out;
}
Tensor cpu_mul_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), dt, [=](double v){ return v * s; });
    });
    return out;
}
Tensor cpu_div_copy(const Tensor& a, double s) {
    const Dtype dt = a.dtype();
    if (is_integer_dtype(dt) && s == 0.0) throw std::runtime_error("Division by zero");
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), dt, [=](double v){ return v / s; });
    });
    return out;
}

Tensor cpu_sub_copy_scalar_tensor(double s, const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), dt, [=](double v){ return s - v; });
    });
    return out;
}

Tensor cpu_div_copy_scalar_tensor(double s, const Tensor& a) {
    const Dtype dt = a.dtype();
    if (is_integer_dtype(dt)) {
        dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
            const T* p = a.data<T>();
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                if (p[i] == (T)0) throw std::runtime_error("Division by zero");
        });
    }
    Tensor out(a.shape(), a.dtype(), a.device(), a.requires_grad());
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), dt, [=](double v){ return s / v; });
    });
    return out;
}

} // namespace OwnTensor
