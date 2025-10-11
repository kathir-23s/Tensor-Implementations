#include "../include/UnaryOps.h"
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <cmath>
#include<cstddef>
#include<omp.h>
using namespace std;
//out of place dispatcher
template <typename Func>
void dispatch_dtype(const Tensor& in, Tensor& out, Func func) {
size_t n = in.numel();

switch (in.dtype()) {
case Dtype::Int16:
func(static_cast<const int16_t*>(in.data()), static_cast<int16_t*>(out.data()), n);
break;
case Dtype::Int32:
func(static_cast<const int32_t*>(in.data()), static_cast<int32_t*>(out.data()), n);
break;
case Dtype::Int64:
func(static_cast<const int64_t*>(in.data()), static_cast<int64_t*>(out.data()), n);
break;
case Dtype::Bfloat16:
case Dtype::Float16:
func(static_cast<const uint16_t*>(in.data()), static_cast<uint16_t*>(out.data()), n);
break;
case Dtype::Float32:
func(static_cast<const float*>(in.data()), static_cast<float*>(out.data()), n);
break;
case Dtype::Float64:
func(static_cast<const double*>(in.data()), static_cast<double*>(out.data()), n);
break;
default:
throw std::runtime_error("Unsupported dtype in out-of-place operation");
}
}

//in place dispatcher
template <typename Func>
void dispatch_dtype_inplace(Tensor& t, Func func) {
size_t n = t.numel();

switch (t.dtype()) {
case Dtype::Int16:
func(static_cast<int16_t*>(t.data()), n);
break;
case Dtype::Int32:
func(static_cast<int32_t*>(t.data()), n);
break;
case Dtype::Int64:
func(static_cast<int64_t*>(t.data()), n);
break;
case Dtype::Bfloat16:
case Dtype::Float16:
func(static_cast<uint16_t*>(t.data()), n);
break;
case Dtype::Float32:
func(static_cast<float*>(t.data()), n);
break;
case Dtype::Float64:
func(static_cast<double*>(t.data()), n);
break;
default:
throw std::runtime_error("Unsupported dtype in in-place operation");
}
}

//INPLACE OPERATIONS:
//SQUARE
template <typename T>
void square_impl(T* data, size_t n) {
// Operation: data[i] = data[i] * data[i]
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
data[i] *= data[i];
}
}

//square root
template <typename T>
void sqrt_impl(T* data, size_t n) {
// Operation: output[i] = sqrt(input[i])
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
data[i] = std::sqrt(data[i]);
}
}

//power
template <typename T, typename P> // P is the exponent type
void power_impl(T* data,size_t n, P exponent) {
// Operation: output[i] = pow(input[i], exponent)
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
data[i] = std::pow(data[i], exponent);
}
}

//negate
template <typename T>
void negate_impl(T* data, size_t n) {
// Operation: data[i] = -data[i]
// Note: For custom 16-bit floats stored as uint16_t, this requires
// specialized bitwise negation in a real library. Here we use - for concept.
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
data[i] = -data[i];
}
}

//abs
template <typename T>
void abs_impl(T* data, size_t n) {
// Operation: data[i] = |data[i]|
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
data[i] = std::abs(data[i]);
}
}

//sign
template <typename T>
void sign_impl(T* data, size_t n) {
// Operation: data[i] = sign(data[i])
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) { 
data[i] = (data[i] > 0) - (data[i] < 0); // Results in 1, -1, or 0
}
}

//RECIPROCAL
template <typename T>
void reciprocal_impl(T* data, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
if (data[i] == 0) {
data[i] = 0; // Handle division by zero gracefully
continue;
}
data[i] = 1 / data[i];
}
}

//INPLACE USER FN CALL DEFINITIONS:
//SQUARE
void square_(Tensor& t) {
dispatch_dtype_inplace(t, [](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
square_impl<T>(data, n);
});
}

//SQUARE ROOT
void square_root_(Tensor& t) {
dispatch_dtype_inplace(t, [](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
sqrt_impl<T>(data, n);
});
}

//POWER
void power_(Tensor& t, int exponent) {
dispatch_dtype_inplace(t, [exponent](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
power_impl<T, int>(data, n, exponent);
});
}

//NEGATE
void negator_(Tensor& t) {
dispatch_dtype_inplace(t, [](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
negate_impl<T>(data, n);
});
}

//ABS
void absolute_(Tensor& t) {
dispatch_dtype_inplace(t, [](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
abs_impl<T>(data, n);
});
}

//SIGN
void sign_(Tensor& t) {
dispatch_dtype_inplace(t, [](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
sign_impl<T>(data, n);
});
}

//reciprocal
void reciprocal_(Tensor& t) {
dispatch_dtype_inplace(t, [](auto* data, size_t n) {
using T = std::remove_pointer_t<decltype(data)>;
reciprocal_impl<T>(data, n);
});
}

//OUT OF PLACE OPERATIONS:
//SQUARE
template <typename T>
void square2_impl(const T* in, T* out, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
out[i] = in[i] * in[i];
}
} 

//square root
template <typename T>
void sqrt2_impl(const T* in, T* out, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
out[i] = std::sqrt(in[i]);
}
}

//power
template <typename T, typename P> // P is the exponent type
void power2_impl(const T* in, T* out, size_t n, P exponent) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
out[i] = std::pow(in[i], exponent);
}
}

//negate
template <typename T>
void negate2_impl(const T* in, T* out, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
out[i] = -in[i];
}
}

//abs
template <typename T>
void abs2_impl(const T* in, T* out, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
out[i] = std::abs(in[i]);
}
}

//sign
template <typename T>
void sign2_impl(const T* in, T* out, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) { 
out[i] = (in[i] > 0) - (in[i] < 0); // Results in 1, -1, or 0
}
}

//reciprocal
template <typename T>
void reciprocal2_impl(const T* in, T* out, size_t n) {
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
if (in[i] == 0) {
out[i] = 0; // Handle division by zero gracefully
continue;
}
out[i] = 1 / in[i];
}
}

//OUT OF PLACE USER FN CALL DEFINITIONS:
//SQUARE
Tensor square(const Tensor& t) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
square2_impl<T>(in_data, out_data, n);
});
return out;
}

//SQUARE ROOT
Tensor square_root(const Tensor& t) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
sqrt2_impl<T>(in_data, out_data, n);
});
return out; 
}

//POWER
Tensor power(const Tensor& t, int exponent) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [exponent](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
power2_impl<T, int>(in_data, out_data, n, exponent);
});
return out;
}

//NEGATE
Tensor negator(const Tensor& t) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
negate2_impl<T>(in_data, out_data, n);
});
return out;
}

//ABS
Tensor absolute(const Tensor& t) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
abs2_impl<T>(in_data, out_data, n);
});
return out;
}

//sign
Tensor sign(const Tensor& t) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
sign2_impl<T>(in_data, out_data, n);
});
return out;
}

//reciprocal
Tensor reciprocal(const Tensor& t) {
Tensor out({t.shape()}, t.dtype(), t.device(), t.requires_grad());
dispatch_dtype(t, out, [](auto in_data, auto out_data, size_t n) {
using T = std::remove_const_t<std::remove_pointer_t<decltype(in_data)>>;
reciprocal2_impl<T>(in_data, out_data, n);
});
return out;
}
