#pragma once
#include "Tensor.h"

namespace OwnTensor{
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
Tensor operator+(const Tensor& tensor, T scalar);

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
Tensor operator+(T scalar, const Tensor& tensor);
}