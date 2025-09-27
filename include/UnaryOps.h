#pragma once
#include "Tensor.h"
//=======================================//
//========== Basic Arithmetics ==========//
//=======================================//
























//=======================================//
//============ Trigonometric ============//
//=======================================//


























//=======================================//
//===== exponentials and logarithms =====//
//=======================================//

// Operation type for dispatch
enum class OpType { Exp = 0, Exp2, Cexp };

// Type-erased kernel function pointer
using KernelFn = void(*)(const void*, void*, size_t);

// Kernel function templates
template<typename T, T(*Func)(T)>
void unary_kernel(const T* in, T* out, size_t n);

// Function wrappers for exp, exp2
float expf_wrap(float x);
double exp_wrap(double x);
float exp2f_wrap(float x);
double exp2_wrap(double x);

// Kernel instantiations
void exp_float_kernel(const void* in, void* out, size_t n);
void exp_double_kernel(const void* in, void* out, size_t n);
void exp2_float_kernel(const void* in, void* out, size_t n);
void exp2_double_kernel(const void* in, void* out, size_t n);

// Kernel dispatch table
constexpr size_t DTYPE_COUNT = 2; // float, double
constexpr size_t OPTYPE_COUNT = 2; // exp, exp2
extern const KernelFn kernel_table[OPTYPE_COUNT][DTYPE_COUNT];

// Helper to map Tensor dtype to index
size_t dtype_index(Dtype dtype);

// Dispatch function
Tensor unary_dispatch(const Tensor& input, OpType op);

// User API
Tensor exp(const Tensor& input);      // out-of-place
Tensor exp2(const Tensor& input);     // out-of-place
void exp_(Tensor& input);             // in-place
void exp2_(Tensor& input);            // in-place






















//=======================================//
//=========== Data Operations ===========//
//=======================================//