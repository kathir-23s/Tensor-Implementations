#pragma once
#include "core/Tensor.h"

namespace OwnTensor {

// In-place operators
template<typename S> Tensor& operator+=(Tensor& tensor, S scalar);
template<typename S> Tensor& operator-=(Tensor& tensor, S scalar);
template<typename S> Tensor& operator*=(Tensor& tensor, S scalar);
template<typename S> Tensor& operator/=(Tensor& tensor, S scalar);

// Tensor (lhs) ⊗ Scalar (rhs)
template<typename S> Tensor operator+(const Tensor& tensor, S scalar);
template<typename S> Tensor operator-(const Tensor& tensor, S scalar);
template<typename S> Tensor operator*(const Tensor& tensor, S scalar);
template<typename S> Tensor operator/(const Tensor& tensor, S scalar);

// Scalar (lhs) ⊗ Tensor (rhs)
template<typename S> Tensor operator+(S scalar, const Tensor& tensor);
template<typename S> Tensor operator-(S scalar, const Tensor& tensor);
template<typename S> Tensor operator*(S scalar, const Tensor& tensor);
template<typename S> Tensor operator/(S scalar, const Tensor& tensor);

//Scalar comparisons
template<typename S> Tensor operator==(const Tensor& t, S scalar);
template<typename S> Tensor operator!=(const Tensor& t, S scalar);
template<typename S> Tensor operator<=(const Tensor& t, S scalar);
template<typename S> Tensor operator>(const Tensor& t, S scalar);
template<typename S> Tensor operator>=(const Tensor& t, S scalar);
template<typename S> Tensor operator<(const Tensor& t, S scalar);

template<typename S> Tensor operator==( S scalar, const Tensor& t);
template<typename S> Tensor operator!=( S scalar, const Tensor& t);
template<typename S> Tensor operator<=( S scalar, const Tensor& t);
template<typename S> Tensor operator> ( S scalar, const Tensor& t);
template<typename S> Tensor operator>=( S scalar, const Tensor& t);
template<typename S> Tensor operator< ( S scalar, const Tensor& t);

//Logical operations with scalars = throws error
template<typename S> Tensor logical_AND(const Tensor& a, S b);
template<typename S> Tensor logical_OR(const Tensor& a, S b);
template<typename S> Tensor logical_XOR(const Tensor& a, S b);
template<typename S> Tensor logical_NOT(S a);

template<typename S>Tensor logical_AND(S a, const Tensor& b);
template<typename S>Tensor logical_OR(S a, const Tensor& b);
template<typename S>Tensor logical_XOR(S a, const Tensor& b);   


} // namespace OwnTensor
