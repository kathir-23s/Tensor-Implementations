#pragma once

#include "core/Tensor.h"
#ifdef WITH_CUDA//✨✨✨
#include <driver_types.h>
#endif//✨✨✨


namespace OwnTensor{
// exponentials and logarithms
// ============================================================
// Out-of-place unary trigonometric functions
// ============================================================
Tensor exp(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor exp2(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor log(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor log2(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor log10(const Tensor& input, cudaStream_t stream = 0);//✨✨✨

// ============================================================
// In-place unary trigonometric functions
// ============================================================
void exp_(Tensor& input, cudaStream_t stream = 0);//✨✨✨
void exp2_(Tensor& input, cudaStream_t stream = 0);//✨✨✨
void log_(Tensor& input, cudaStream_t stream = 0);//✨✨✨
void log2_(Tensor& input, cudaStream_t stream = 0);//✨✨✨
void log10_(Tensor& input, cudaStream_t stream = 0);//✨✨✨
} // end of namespace