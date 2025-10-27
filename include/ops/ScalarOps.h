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

} // namespace OwnTensor
