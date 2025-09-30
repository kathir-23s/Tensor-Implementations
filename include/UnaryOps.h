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

// Table of kernels per dtype for each op
enum class OpType { Exp = 0, Exp2, Log, Log2, Log10};

// Type-erased kernel function pointer
using KernelFn = void(*)(const void*, void*, size_t);

// Kernel function templates
template<typename T, T(*Func)(T)>
void unary_kernel(const T* in, T* out, size_t n);

// Function wrappers for exp, exp2, log, log2, log10
float expf_wrap(float x);
double exp_wrap(double x);
float exp2f_wrap(float x);
double exp2_wrap(double x);
float logf_wrap(float x);
double log_wrap(double x);
float log2f_wrap(float x);
double log2_wrap(double x);
float log10f_wrap(float x);
double log10_wrap(double x);

// Kernel instantiations
void exp_float_kernel(const void* in, void* out, size_t n);
void exp_double_kernel(const void* in, void* out, size_t n);
void exp2_float_kernel(const void* in, void* out, size_t n);
void exp2_double_kernel(const void* in, void* out, size_t n);
void log_float_kernel(const void* in, void* out, size_t n);
void log_double_kernel(const void* in, void* out, size_t n);
void log2_float_kernel(const void* in, void* out, size_t n);
void log2_double_kernel(const void* in, void* out, size_t n);
void log10_float_kernel(const void* in, void* out, size_t n);
void log10_double_kernel(const void* in, void* out, size_t n);

// Kernel dispatch table
constexpr size_t DTYPE_COUNT = 2; // float, double
constexpr size_t OPTYPE_COUNT = 5; // exp, exp2, log, log2, log10
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

Tensor log(const Tensor& input);      // out-of-place
Tensor log2(const Tensor& input);     // out-of-place
Tensor log10(const Tensor& input);     // out-of-place
void log_(Tensor& input);             // in-place
void log2_(Tensor& input);            // in-place
void log10_(Tensor& input);            // in-place

















//=======================================//
//=========== Data Operations ===========//
//=======================================//