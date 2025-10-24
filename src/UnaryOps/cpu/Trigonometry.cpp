#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"

namespace OwnTensor {
using namespace OwnTensor::detail;

// --------------------------------------------------------------------
// Trig selector + utilities
// --------------------------------------------------------------------
enum class Trig {
    Sin, Cos, Tan, Asin, Acos, Atan,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
};

static inline void*       get_data_mut(Tensor& t)   { return t.data(); }
static inline const void* get_data(const Tensor& t) { return t.data(); }
static inline Tensor      make_like(const Tensor& x){ return Tensor(Shape{x.shape()}, x.dtype(), x.device(), x.requires_grad()); }

template <typename T, Trig K>
static inline T trig_apply_scalar(T x) {
    if constexpr (K == Trig::Sin)   { using std::sin;   return static_cast<T>(sin(x));   }
    if constexpr (K == Trig::Cos)   { using std::cos;   return static_cast<T>(cos(x));   }
    if constexpr (K == Trig::Tan)   { using std::tan;   return static_cast<T>(tan(x));   }
    if constexpr (K == Trig::Asin)  { using std::asin;  return static_cast<T>(asin(x));  }
    if constexpr (K == Trig::Acos)  { using std::acos;  return static_cast<T>(acos(x));  }
    if constexpr (K == Trig::Atan)  { using std::atan;  return static_cast<T>(atan(x));  }
    if constexpr (K == Trig::Sinh)  { using std::sinh;  return static_cast<T>(sinh(x));  }
    if constexpr (K == Trig::Cosh)  { using std::cosh;  return static_cast<T>(cosh(x));  }
    if constexpr (K == Trig::Tanh)  { using std::tanh;  return static_cast<T>(tanh(x));  }
    if constexpr (K == Trig::Asinh) { using std::asinh; return static_cast<T>(asinh(x)); }
    if constexpr (K == Trig::Acosh) { using std::acosh; return static_cast<T>(acosh(x)); }
    if constexpr (K == Trig::Atanh) { using std::atanh; return static_cast<T>(atanh(x)); }
    return T(0);
}

// --------------------------------------------------------------------
// In-place
// --------------------------------------------------------------------
template <Trig K>
static inline void unary_inplace_impl(Tensor& x) {
    if (!x.is_contiguous()) throw std::runtime_error("Non-contiguous tensors not supported yet.");
    const size_t n = x.numel();

    dispatch_by_dtype(x.dtype(), [&](auto tag){
        using Tag = std::decay_t<decltype(tag)>;

        if constexpr (std::is_same_v<Tag, float>) {
            auto* p = reinterpret_cast<float*>(get_data_mut(x));
            for (size_t i = 0; i < n; ++i) p[i] = trig_apply_scalar<float, K>(p[i]);
        }
        else if constexpr (std::is_same_v<Tag, double>) {
            auto* p = reinterpret_cast<double*>(get_data_mut(x));
            for (size_t i = 0; i < n; ++i) p[i] = trig_apply_scalar<double, K>(p[i]);
        }
        else if constexpr (std::is_same_v<Tag, float16_t>) {
            auto* p = reinterpret_cast<float16_t*>(get_data_mut(x));
            for (size_t i = 0; i < n; ++i) {
                float f = static_cast<float>(p[i]);
                p[i] = float16_t(trig_apply_scalar<float, K>(f));
            }
        }
        else if constexpr (std::is_same_v<Tag, bfloat16_t>) {
            auto* p = reinterpret_cast<bfloat16_t*>(get_data_mut(x));
            for (size_t i = 0; i < n; ++i) {
                float f = static_cast<float>(p[i]);
                p[i] = bfloat16_t(trig_apply_scalar<float, K>(f));
            }
        }
        else if constexpr (std::is_same_v<Tag, int16_t> ||
                           std::is_same_v<Tag, int32_t> ||
                           std::is_same_v<Tag, int64_t>) {
            throw std::runtime_error("In-place trig ops not supported for integer tensors. Use out-of-place.");
        }
        else {
            static_assert(!sizeof(Tag*), "Unsupported dtype");
        }
    });
}

// --------------------------------------------------------------------
// Out-of-place
// --------------------------------------------------------------------
template <Trig K>
static inline Tensor unary_out_impl(const Tensor& x) {
    if (!x.is_contiguous()) throw std::runtime_error("Non-contiguous tensors not supported yet.");
    const size_t n = x.numel();

    return dispatch_by_dtype(x.dtype(), [&](auto tag){
        using Tag = std::decay_t<decltype(tag)>;

        if constexpr (std::is_same_v<Tag, float>) {
            Tensor y = make_like(x);
            const float* in = reinterpret_cast<const float*>(get_data(x));
            float* out = reinterpret_cast<float*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) out[i] = trig_apply_scalar<float, K>(in[i]);
            return y;
        }
        else if constexpr (std::is_same_v<Tag, double>) {
            Tensor y = make_like(x);
            const double* in = reinterpret_cast<const double*>(get_data(x));
            double* out = reinterpret_cast<double*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) out[i] = trig_apply_scalar<double, K>(in[i]);
            return y;
        }
        else if constexpr (std::is_same_v<Tag, float16_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float16, x.device(), x.requires_grad()});
            const auto* in = reinterpret_cast<const float16_t*>(get_data(x));
            auto* out = reinterpret_cast<float16_t*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) {
                float f = static_cast<float>(in[i]);
                out[i] = float16_t(trig_apply_scalar<float, K>(f));
            }
            return y;
        }
        else if constexpr (std::is_same_v<Tag, bfloat16_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Bfloat16, x.device(), x.requires_grad()});
            const auto* in = reinterpret_cast<const bfloat16_t*>(get_data(x));
            auto* out = reinterpret_cast<bfloat16_t*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) {
                float f = static_cast<float>(in[i]);
                out[i] = bfloat16_t(trig_apply_scalar<float, K>(f));
            }
            return y;
        }
        else if constexpr (std::is_same_v<Tag, int16_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float32, x.device(), x.requires_grad()});
            const int16_t* in = reinterpret_cast<const int16_t*>(get_data(x));
            float* out = reinterpret_cast<float*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) out[i] = trig_apply_scalar<float, K>(static_cast<float>(in[i]));
            return y;
        }
        else if constexpr (std::is_same_v<Tag, int32_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float32, x.device(), x.requires_grad()});
            const int32_t* in = reinterpret_cast<const int32_t*>(get_data(x));
            float* out = reinterpret_cast<float*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) out[i] = trig_apply_scalar<float, K>(static_cast<float>(in[i]));
            return y;
        }
        else if constexpr (std::is_same_v<Tag, int64_t>) {
            Tensor y(Shape{x.shape()}, TensorOptions{Dtype::Float64, x.device(), x.requires_grad()});
            const int64_t* in = reinterpret_cast<const int64_t*>(get_data(x));
            double* out = reinterpret_cast<double*>(get_data_mut(y));
            for (size_t i = 0; i < n; ++i) out[i] = trig_apply_scalar<double, K>(static_cast<double>(in[i]));
            return y;
        }
        else {
            static_assert(!sizeof(Tag*), "Unsupported dtype");
        }
    });
}

// --------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------
Tensor sin_cpu  (const Tensor& x){ return unary_out_impl<Trig::Sin>(x); }   void sin__cpu  (Tensor& x){ unary_inplace_impl<Trig::Sin>(x); }
Tensor cos_cpu  (const Tensor& x){ return unary_out_impl<Trig::Cos>(x); }   void cos__cpu  (Tensor& x){ unary_inplace_impl<Trig::Cos>(x); }
Tensor tan_cpu  (const Tensor& x){ return unary_out_impl<Trig::Tan>(x); }   void tan__cpu  (Tensor& x){ unary_inplace_impl<Trig::Tan>(x); }
Tensor asin_cpu (const Tensor& x){ return unary_out_impl<Trig::Asin>(x);}   void asin__cpu (Tensor& x){ unary_inplace_impl<Trig::Asin>(x);} 
Tensor acos_cpu (const Tensor& x){ return unary_out_impl<Trig::Acos>(x);}   void acos__cpu (Tensor& x){ unary_inplace_impl<Trig::Acos>(x);} 
Tensor atan_cpu (const Tensor& x){ return unary_out_impl<Trig::Atan>(x);}   void atan__cpu (Tensor& x){ unary_inplace_impl<Trig::Atan>(x);} 
Tensor sinh_cpu (const Tensor& x){ return unary_out_impl<Trig::Sinh>(x);}   void sinh__cpu (Tensor& x){ unary_inplace_impl<Trig::Sinh>(x);} 
Tensor cosh_cpu (const Tensor& x){ return unary_out_impl<Trig::Cosh>(x);}   void cosh__cpu (Tensor& x){ unary_inplace_impl<Trig::Cosh>(x);} 
Tensor tanh_cpu (const Tensor& x){ return unary_out_impl<Trig::Tanh>(x);}   void tanh__cpu (Tensor& x){ unary_inplace_impl<Trig::Tanh>(x);} 
Tensor asinh_cpu(const Tensor& x){ return unary_out_impl<Trig::Asinh>(x);}  void asinh__cpu(Tensor& x){ unary_inplace_impl<Trig::Asinh>(x);} 
Tensor acosh_cpu(const Tensor& x){ return unary_out_impl<Trig::Acosh>(x);}  void acosh__cpu(Tensor& x){ unary_inplace_impl<Trig::Acosh>(x);} 
Tensor atanh_cpu(const Tensor& x){ return unary_out_impl<Trig::Atanh>(x);}  void atanh__cpu(Tensor& x){ unary_inplace_impl<Trig::Atanh>(x);} 
} // namespace OwnTensor
