
#include <cmath>
#include "../include/Tensor.h"
using namespace std;


// ================================================================================================ //
// ================================== Exponential functions ======================================= //
// ================================================================================================ //

// Helper kernels for exp, exp2
template<typename T, T(*Func)(T)>
void unary_kernel(const T* in, T* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = Func(in[i]);
    }
}

// Function wrappers for exp, exp2
float expf_wrap(float x) { return expf(x); }
double exp_wrap(double x) { return exp(x); }
float exp2f_wrap(float x) { return exp2f(x); }
double exp2_wrap(double x) { return exp2(x); }

// Type-erased kernel function pointer
using KernelFn = void(*)(const void*, void*, size_t);

// Kernel instantiations
void exp_float_kernel(const void* in, void* out, size_t n) {
    unary_kernel<float, expf_wrap>(static_cast<const float*>(in), static_cast<float*>(out), n);
}
void exp_double_kernel(const void* in, void* out, size_t n) {
    unary_kernel<double, exp_wrap>(static_cast<const double*>(in), static_cast<double*>(out), n);
}
void exp2_float_kernel(const void* in, void* out, size_t n) {
    unary_kernel<float, exp2f_wrap>(static_cast<const float*>(in), static_cast<float*>(out), n);
}
void exp2_double_kernel(const void* in, void* out, size_t n) {
    unary_kernel<double, exp2_wrap>(static_cast<const double*>(in), static_cast<double*>(out), n);
}

// Table of kernels per dtype for each op
enum class OpType { Exp = 0, Exp2 };

constexpr size_t DTYPE_COUNT = 2; // float, double
constexpr size_t OPTYPE_COUNT = 2; // exp, exp2

// Dtype indices: 0=float, 1=double
static const KernelFn kernel_table[OPTYPE_COUNT][DTYPE_COUNT] = {
    // Exp
    { exp_float_kernel, exp_double_kernel },
    // Exp2
    { exp2_float_kernel, exp2_double_kernel }
};

// Helper to map Tensor dtype to index
size_t dtype_index(Dtype dtype) {
    switch (dtype) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
        case Dtype::Bfloat16:
        case Dtype::Float16:
        case Dtype::Float32: return 0;
        case Dtype::Float64: return 1;
        default: throw std::runtime_error("Unsupported dtype for exp ops");
    }
}

// Out-of-place operation: returns new tensor
Tensor unary_dispatch(const Tensor& input, OpType op) {
    Tensor output(Shape{input.shape()}, input.dtype(), input.device(), false);
    size_t total_elems = input.numel();
    size_t idx = dtype_index(input.dtype());
    KernelFn fn = kernel_table[static_cast<size_t>(op)][idx];
    fn(input.data(), output.data(), total_elems);
    return output;
}

Tensor exp(const Tensor& input) {
    return unary_dispatch(input, OpType::Exp);
}

Tensor exp2(const Tensor& input) {
    return unary_dispatch(input, OpType::Exp2);
}

// In-place operation: modifies input tensor
void exp_(Tensor& input) {
    size_t total_elems = input.numel();
    size_t idx = dtype_index(input.dtype());
    KernelFn fn = kernel_table[static_cast<size_t>(OpType::Exp)][idx];
    fn(input.data(), input.data(), total_elems);
}

void exp2_(Tensor& input) {
    size_t total_elems = input.numel();
    size_t idx = dtype_index(input.dtype());
    KernelFn fn = kernel_table[static_cast<size_t>(OpType::Exp2)][idx];
    fn(input.data(), input.data(), total_elems);
}



// ================================================================================================ //
// ================================== Logarithmic functions ======================================= //
// ================================================================================================ //