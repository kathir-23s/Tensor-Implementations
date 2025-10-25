#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
// ============================================================
// Out-of-place unary Arithmetics functions
// ============================================================
Tensor square(const Tensor& t);
Tensor square_root(const Tensor& t);
//Tensor power(const Tensor& t, int exponent);
Tensor negator(const Tensor& t); 
Tensor absolute(const Tensor& t);
Tensor sign(const Tensor& t);
Tensor reciprocal(const Tensor& t);
// ============================================================
// In-place unary Arithmetics functions
// ============================================================
void square_(Tensor& t);
void square_root_(Tensor& t);
//void power_(Tensor& t, int exponent);
void negator_(Tensor& t); 
void absolute_(Tensor& t); 
void sign_(Tensor& t);
void reciprocal_(Tensor& t);

} // end of namespace