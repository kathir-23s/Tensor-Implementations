#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
    Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    Tensor operator-(const Tensor& lhs, const Tensor& rhs);
    Tensor operator*(const Tensor& lhs, const Tensor& rhs);
    Tensor operator/(const Tensor& lhs, const Tensor& rhs);

    Tensor operator+=(Tensor& lhs, const Tensor& rhs);
    Tensor operator-=(Tensor& lhs, const Tensor& rhs);
    Tensor operator*=(Tensor& lhs, const Tensor& rhs);
    Tensor operator/=(Tensor& lhs, const Tensor& rhs);

    // Element-wise comparisons (return Bool tensors)
    Tensor operator==(const Tensor& lhs, const Tensor& rhs);
    Tensor operator!=(const Tensor& lhs, const Tensor& rhs);
    Tensor operator<=(const Tensor& lhs, const Tensor& rhs);
    Tensor operator>=(const Tensor& lhs, const Tensor& rhs);
    Tensor operator<(const Tensor& lhs, const Tensor& rhs);
    Tensor operator>(const Tensor& lhs, const Tensor& rhs);

    //Logical operations
    // Element-wise logical operations
    Tensor logical_AND(const Tensor& a, const Tensor& b);
    Tensor logical_OR(const Tensor& a, const Tensor& b);
    Tensor logical_XOR(const Tensor& a, const Tensor& b);
    Tensor logical_NOT(const Tensor& a);
    
    
}