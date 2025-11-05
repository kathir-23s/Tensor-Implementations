#pragma once

#include "core/Tensor.h"
#include "ops/helpers/Trigonometry.hpp"

#ifdef WITH_CUDA //✨✨✨
    #include <driver_types.h>
#endif //✨✨✨

namespace OwnTensor{
// ============================================================
// Out-of-place trigonometric functions
// ============================================================

// Basic trigonometric functions
Tensor sin(const Tensor& input, cudaStream_t stream = 0);
Tensor cos(const Tensor& input, cudaStream_t stream = 0);
Tensor tan(const Tensor& input, cudaStream_t stream = 0);

// Inverse trigonometric functions
Tensor asin(const Tensor& input, cudaStream_t stream = 0);
Tensor acos(const Tensor& input, cudaStream_t stream = 0);
Tensor atan(const Tensor& input, cudaStream_t stream = 0);

// Hyperbolic functions
Tensor sinh(const Tensor& input, cudaStream_t stream = 0);
Tensor cosh(const Tensor& input, cudaStream_t stream = 0);
Tensor tanh(const Tensor& input, cudaStream_t stream = 0);

// Inverse hyperbolic functions
Tensor asinh(const Tensor& input, cudaStream_t stream = 0);
Tensor acosh(const Tensor& input, cudaStream_t stream = 0);
Tensor atanh(const Tensor& input, cudaStream_t stream = 0);

// ============================================================
// In-place trigonometric functions
// ============================================================

// Basic trigonometric functions
void sin_(Tensor& input, cudaStream_t stream = 0);
void cos_(Tensor& input, cudaStream_t stream = 0);
void tan_(Tensor& input, cudaStream_t stream = 0);

// Inverse trigonometric functions
void asin_(Tensor& input, cudaStream_t stream = 0);
void acos_(Tensor& input, cudaStream_t stream = 0);
void atan_(Tensor& input, cudaStream_t stream = 0);

// Hyperbolic functions
void sinh_(Tensor& input, cudaStream_t stream = 0);
void cosh_(Tensor& input, cudaStream_t stream = 0);
void tanh_(Tensor& input, cudaStream_t stream = 0);

// Inverse hyperbolic functions
void asinh_(Tensor& input, cudaStream_t stream = 0);
void acosh_(Tensor& input, cudaStream_t stream = 0);
void atanh_(Tensor& input, cudaStream_t stream = 0);

} // namespace OwnTensor