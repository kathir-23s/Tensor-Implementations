#pragma once

#include "Tensor.h" 
#include "Types.h"  // your Tensor, Dtype, Device, TensorOptions
#include <cstdint>


namespace OwnTensor
{
	enum class UnaryOp : uint16_t {
	// Trig + Hyperbolic
	Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
	// Exponentials / Logs
	Exp, Exp2, Log, Log2, Log10,
	// Algebraic
	Square, Sqrt, Neg, Abs, Sign, Reciprocal,
	// Parametric
	Power
	};

	// Entry points (non-mutating / in-place)
	Tensor unary       (const Tensor& x, UnaryOp op);
	void   unary_      (Tensor& x,      UnaryOp op);
	Tensor unary_param (const Tensor& x, UnaryOp op, double p);
	void   unary_param_(Tensor& x,       UnaryOp op, double p);

	// Optional: convenience wrappers to match old names
	Tensor sin(const Tensor& x);     Tensor cos(const Tensor& x);     Tensor tan(const Tensor& x);
	Tensor asin(const Tensor& x);    Tensor acos(const Tensor& x);    Tensor atan(const Tensor& x);
	Tensor sinh(const Tensor& x);    Tensor cosh(const Tensor& x);    Tensor tanh(const Tensor& x);
	Tensor asinh(const Tensor& x);   Tensor acosh(const Tensor& x);   Tensor atanh(const Tensor& x);

	Tensor exp(const Tensor& x);     Tensor exp2(const Tensor& x);
	Tensor log(const Tensor& x);     Tensor log2(const Tensor& x);    Tensor log10(const Tensor& x);

	Tensor square(const Tensor& x);  Tensor square_root(const Tensor& x);
	Tensor negator(const Tensor& x); Tensor absolute(const Tensor& x);
	Tensor sign(const Tensor& x);    Tensor reciprocal(const Tensor& x);
	Tensor power(const Tensor& x, int e);

	void sin_(Tensor& x);     void cos_(Tensor& x);     void tan_(Tensor& x);
	void asin_(Tensor& x);    void acos_(Tensor& x);    void atan_(Tensor& x);
	void sinh_(Tensor& x);    void cosh_(Tensor& x);    void tanh_(Tensor& x);
	void asinh_(Tensor& x);   void acosh_(Tensor& x);   void atanh_(Tensor& x);

	void exp_(Tensor& x);     void exp2_(Tensor& x);
	void log_(Tensor& x);     void log2_(Tensor& x);    void log10_(Tensor& x);

	void square_(Tensor& x);  void square_root_(Tensor& x);
	void negator_(Tensor& x); void absolute_(Tensor& x);
	void sign_(Tensor& x);    void reciprocal_(Tensor& x);
	void power_(Tensor& x, int e);
}