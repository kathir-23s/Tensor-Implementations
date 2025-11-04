#pragma once
#include "core/Tensor.h"
#include <driver_types.h>//✨✨✨
namespace OwnTensor {
// ============================================================
// Out-of-place unary Arithmetics functions
// ============================================================
Tensor square(const Tensor& t, cudaStream_t stream);//✨✨✨
Tensor sqrt(const Tensor& t, cudaStream_t stream);//✨✨✨
Tensor neg(const Tensor& t, cudaStream_t stream); //✨✨✨
Tensor abs(const Tensor& t, cudaStream_t stream);//✨✨✨
Tensor sign(const Tensor& t, cudaStream_t stream);//✨✨✨
Tensor reciprocal(const Tensor& t, cudaStream_t stream);//✨✨✨
// ============================================================
// In-place unary Arithmetics functions
// ============================================================
void square_(Tensor& t, cudaStream_t stream);//✨✨✨
void sqrt_(Tensor& t, cudaStream_t stream);//✨✨✨
void neg_(Tensor& t, cudaStream_t stream); //✨✨✨
void abs_(Tensor& t, cudaStream_t stream); //✨✨✨
void sign_(Tensor& t, cudaStream_t stream);//✨✨✨
void reciprocal_(Tensor& t, cudaStream_t stream);//✨✨✨

// Out-of-place power functions
Tensor pow(const Tensor& t, int exponent, cudaStream_t stream);//✨✨✨
Tensor pow(const Tensor& t, float exponent, cudaStream_t stream);//✨✨✨
Tensor pow(const Tensor& t, double exponent, cudaStream_t stream);//✨✨✨

// In-place power functions
void pow_(Tensor& t, int exponent, cudaStream_t stream);//✨✨✨
void pow_(Tensor& t, float exponent, cudaStream_t stream);//✨✨✨
void pow_(Tensor& t, double exponent, cudaStream_t stream);
} // end of namespace