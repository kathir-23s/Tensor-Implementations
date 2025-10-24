#pragma once

#include "core/Tensor.h"

namespace OwnTensor {

    // In-place operators
    template<typename T>
    Tensor& operator+=(Tensor& tensor, T scalar);
    
    template<typename T>
    Tensor& operator-=(Tensor& tensor, T scalar);
    
    template<typename T>
    Tensor& operator*=(Tensor& tensor, T scalar);
    
    template<typename T>
    Tensor& operator/=(Tensor& tensor, T scalar);

    // Binary operators (Tensor + Scalar)
    template<typename T>
    Tensor operator+(const Tensor& tensor, T scalar);
    
    template<typename T>
    Tensor operator-(const Tensor& tensor, T scalar);
    
    template<typename T>
    Tensor operator*(const Tensor& tensor, T scalar);
    
    template<typename T>
    Tensor operator/(const Tensor& tensor, T scalar);

    // Binary operators (Scalar + Tensor)
    template<typename T>
    Tensor operator+(T scalar, const Tensor& tensor);
    
    template<typename T>
    Tensor operator-(T scalar, const Tensor& tensor);
    
    template<typename T>
    Tensor operator*(T scalar, const Tensor& tensor);
    
    template<typename T>
    Tensor operator/(T scalar, const Tensor& tensor);

} // namespace OwnTensor