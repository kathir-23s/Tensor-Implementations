#pragma once

#include "core/Tensor.h"
namespace OwnTensor{
// exponentials and logarithms
// ============================================================
// Out-of-place unary trigonometric functions
// ============================================================
Tensor exp(const Tensor& input);
Tensor exp2(const Tensor& input);
Tensor log(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor log10(const Tensor& input);

// ============================================================
// In-place unary trigonometric functions
// ============================================================
void exp_(Tensor& input);
void exp2_(Tensor& input);
void log_(Tensor& input);
void log2_(Tensor& input);
void log10_(Tensor& input);
} // end of namespace