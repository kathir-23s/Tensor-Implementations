#pragma once
#ifndef CONDITIONAL_OPS_H
#define CONDITIONAL_OPS_H

#include "core/Tensor.h"

namespace OwnTensor {

// Forward declarations for CPU and CUDA backends
void cpu_where(const Tensor& condition, const Tensor& input, 
               const Tensor& other, Tensor& out);

void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out);

// Public API - main where function
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);

// Scalar overloads
Tensor where(const Tensor& condition, double input_scalar, const Tensor& other);
Tensor where(const Tensor& condition, const Tensor& input, double other_scalar);
Tensor where(const Tensor& condition, double input_scalar, double other_scalar);

} // namespace OwnTensor

#endif
