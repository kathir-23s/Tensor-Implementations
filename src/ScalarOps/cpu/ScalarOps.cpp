// ScalarOps.cpp - FIXED DIVISION OPERATORS
#include <cstdint>
#include <stdexcept>
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"  // ✅ For promote_dtypes_division

namespace OwnTensor {
namespace { // file-local helpers

inline bool is_integer_dtype(Dtype dt) {
    return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
}

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

// Special version for comparison ops that write bool
template <typename T, typename F>
inline void apply_copy_to_bool(const T* src, uint8_t* dst, size_t n, Dtype dt, F&& f) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f(ld<T>(src, i, dt)) ? 1 : 0;
    }
}

// ✅ NEW: Helper to determine promoted dtype for division
inline Dtype get_division_output_dtype(Dtype input_dtype) {
    // Bool → Float32
    if (input_dtype == Dtype::Bool) return Dtype::Float32;
    
    // Integer types → Float32
    if (is_integer_dtype(input_dtype)) return Dtype::Float32;
    
    // Float types → Keep same (Float16/BFloat16/Float32/Float64)
    return input_dtype;
}

// ✅ NEW: Template function for cross-type division
template <typename SrcT, typename DstT>
inline void apply_div_cross_type(const SrcT* src, DstT* dst, size_t n, Dtype src_dt, Dtype dst_dt, double s) {
    for (size_t i = 0; i < n; ++i) {
        double val = ld<SrcT>(src, i, src_dt) / s;
        st<DstT>(dst, i, val, dst_dt);
    }
}

} // anon

// --------- Arithmetic ops (unchanged) ---------
// --------- Arithmetic ops (unchanged) ---------
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

// ✅ FIXED: Division in-place (check if promotion needed)
void cpu_div_inplace(Tensor& t, double s) {
    const Dtype dt = t.dtype();
    
    // ✅ Check if this would require promotion
    Dtype promoted_dt = get_division_output_dtype(dt);
    if (promoted_dt != dt) {
        throw std::runtime_error(
            "In-place division /= requires float dtype. "
            "Input is " + get_dtype_name(dt) + " but division needs " + 
            get_dtype_name(promoted_dt) + ". Use regular division (/) instead."
        );
    }
    
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

// ✅ FIXED: Division creates Float32 output for integers/bool
Tensor cpu_div_copy(const Tensor& a, double s) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = get_division_output_dtype(input_dt);
    
    if (is_integer_dtype(input_dt) && s == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    
    // Create output with promoted dtype
    Tensor out(a.shape(), output_dt, a.device(), a.requires_grad());
    
    // If types match, use same-type path
    if (input_dt == output_dt) {
        dispatch_by_dtype(input_dt, [&](auto d){ using T = decltype(d);
            apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), input_dt, 
                         [=](double v){ return v / s; });
        });
    } else {
        // Cross-type conversion (Int16/Bool → Float32)
        dispatch_by_dtype(input_dt, [&](auto d_in){ 
            using SrcT = decltype(d_in);
            dispatch_by_dtype(output_dt, [&](auto d_out){
                using DstT = decltype(d_out);
                apply_div_cross_type<SrcT, DstT>(
                    a.data<SrcT>(), out.data<DstT>(), 
                    a.numel(), input_dt, output_dt, s
                );
            });
        });
    }
    
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

// ✅ FIXED: Scalar / Tensor also promotes to float
Tensor cpu_div_copy_scalar_tensor(double s, const Tensor& a) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = get_division_output_dtype(input_dt);
    
    // Check for division by zero in integer tensors
    if (is_integer_dtype(input_dt)) {
        dispatch_by_dtype(input_dt, [&](auto d){ using T = decltype(d);
            const T* p = a.data<T>();
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                if (p[i] == (T)0) throw std::runtime_error("Division by zero");
        });
    }
    
    Tensor out(a.shape(), output_dt, a.device(), a.requires_grad());
    
    if (input_dt == output_dt) {
        dispatch_by_dtype(input_dt, [&](auto d){ using T = decltype(d);
            apply_copy<T>(a.data<T>(), out.data<T>(), a.numel(), input_dt, 
                         [=](double v){ return s / v; });
        });
    } else {
        dispatch_by_dtype(input_dt, [&](auto d_in){ 
            using SrcT = decltype(d_in);
            dispatch_by_dtype(output_dt, [&](auto d_out){
                using DstT = decltype(d_out);
                const SrcT* src = a.data<SrcT>();
                DstT* dst = out.data<DstT>();
                for (size_t i = 0; i < a.numel(); ++i) {
                    double val = s / ld<SrcT>(src, i, input_dt);
                    st<DstT>(dst, i, val, output_dt);
                }
            });
        });
    }
    
    return out;
}

// --------- Comparison ops (unchanged) ---------
Tensor cpu_eq_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return v == s; });
    });
    return out;
}

Tensor cpu_neq_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return v != s; });
    });
    return out;
}

Tensor cpu_leq_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return v <= s; });
    });
    return out;
}

Tensor cpu_geq_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return v >= s; });
    });
    return out;
}

Tensor cpu_gt_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return v > s; });
    });
    return out;
}

Tensor cpu_lt_copy(const Tensor& a, double s) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return v < s; });
    });
    return out;
}

Tensor cpu_s_leq_copy(double s, const Tensor& a) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return s <= v; });
    });
    return out;
}

Tensor cpu_s_geq_copy(double s, const Tensor& a) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return s >= v; });
    });
    return out;
}

Tensor cpu_s_gt_copy(double s, const Tensor& a) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return s > v; });
    });
    return out;
}

Tensor cpu_s_lt_copy(double s, const Tensor& a) {
    Tensor out(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    const Dtype dt = a.dtype();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    
    dispatch_by_dtype(dt, [&](auto d){ using T = decltype(d);
        apply_copy_to_bool<T>(a.data<T>(), out_ptr, a.numel(), dt, 
                              [=](double v){ return s < v; });
    });
    return out;
}

} // namespace OwnTensor