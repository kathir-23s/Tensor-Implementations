#pragma once
#include "core/Tensor.h"
#include <driver_types.h>//✨✨✨
namespace OwnTensor {
// ============================================================
// Out-of-place unary Arithmetics functions
// ============================================================
Tensor square(const Tensor& t, cudaStream_t stream=0);//✨✨✨
Tensor sqrt(const Tensor& t, cudaStream_t stream=0);//✨✨✨
Tensor neg(const Tensor& t, cudaStream_t stream=0); //✨✨✨
Tensor abs(const Tensor& t, cudaStream_t stream=0);//✨✨✨
Tensor sign(const Tensor& t, cudaStream_t stream=0);//✨✨✨
Tensor reciprocal(const Tensor& t, cudaStream_t stream=0);//✨✨✨
// ============================================================
// In-place unary Arithmetics functions
// ============================================================
void square_(Tensor& t, cudaStream_t stream=0);//✨✨✨
void sqrt_(Tensor& t, cudaStream_t stream=0);//✨✨✨
void neg_(Tensor& t, cudaStream_t stream=0); //✨✨✨
void abs_(Tensor& t, cudaStream_t stream=0); //✨✨✨
void sign_(Tensor& t, cudaStream_t stream=0);//✨✨✨
void reciprocal_(Tensor& t, cudaStream_t stream=0);//✨✨✨

// Out-of-place power functions
Tensor pow(const Tensor& t, int exponent, cudaStream_t stream=0);//✨✨✨
Tensor pow(const Tensor& t, float exponent, cudaStream_t stream=0);//✨✨✨
Tensor pow(const Tensor& t, double exponent, cudaStream_t stream=0);//✨✨✨

// In-place power functions
void pow_(Tensor& t, int exponent, cudaStream_t stream=0);//✨✨✨
void pow_(Tensor& t, float exponent, cudaStream_t stream=0);//✨✨✨
void pow_(Tensor& t, double exponent, cudaStream_t stream=0);
} // end of namespace