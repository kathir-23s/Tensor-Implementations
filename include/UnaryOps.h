#pragma once
#include "Tensor.h"

// basic arithmetics
//out of place
Tensor square(const Tensor& t);
Tensor square_root(const Tensor& t);
Tensor power(const Tensor& t, int exponent);
Tensor negator(const Tensor& t); 
Tensor absolute(const Tensor& t);
Tensor sign(const Tensor& t);
Tensor reciprocal(const Tensor& t);

//in place
void square_(Tensor& t);
void square_root_(Tensor& t);
void power_(Tensor& t, int exponent);
void negator_(Tensor& t); 
void absolute_(Tensor& t); 
void sign_(Tensor& t);
void reciprocal_(Tensor& t);

// trignometric functions

// ============================================================
// Out-of-place unary trigonometric functions
// ============================================================
Tensor sin  (const Tensor& x);
Tensor cos  (const Tensor& x);
Tensor tan  (const Tensor& x);
Tensor asin (const Tensor& x);
Tensor acos (const Tensor& x);
Tensor atan (const Tensor& x);
Tensor sinh (const Tensor& x);
Tensor cosh (const Tensor& x);
Tensor tanh (const Tensor& x);
Tensor asinh(const Tensor& x);
Tensor acosh(const Tensor& x);
Tensor atanh(const Tensor& x);

// ============================================================
// In-place unary trigonometric functions
// ============================================================
void sin_  (Tensor& x);
void cos_  (Tensor& x);
void tan_  (Tensor& x);
void asin_ (Tensor& x);
void acos_ (Tensor& x);
void atan_ (Tensor& x);
void sinh_ (Tensor& x);
void cosh_ (Tensor& x);
void tanh_ (Tensor& x);
void asinh_(Tensor& x);
void acosh_(Tensor& x);
void atanh_(Tensor& x);


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







// reduction operations