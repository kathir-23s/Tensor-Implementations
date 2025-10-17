#pragma once
#include <cuda_runtime.h>
#include "../Tensor.h"

namespace OwnTensor{
Tensor exp_out_cpu_wrap(const Tensor& input_tensor);
void exp_in_cpu_wrap(Tensor& input_tensor);

template<typename T_In, typename T_Out, T_Out(*Func)(T_Out)>
__global__
void unary_kernel_gpu(const T_In* in, T_Out* out, size_t size);

Tensor exp_out_gpu_wrap(const Tensor& input_tensor);
void exp_in_gpu_wrap(Tensor& input_tensor);

Tensor exp2_out_gpu_wrap(const Tensor& input_tensor);
void exp2_in_gpu_wrap(Tensor& input_tensor);

Tensor log_out_gpu_wrap(const Tensor& input_tensor);
void log_in_gpu_wrap(Tensor& input_tensor);

Tensor log2_out_gpu_wrap(const Tensor& input_tensor);
void log2_in_gpu_wrap(Tensor& input_tensor);

Tensor log10_out_gpu_wrap(const Tensor& input_tensor);
void log10_in_gpu_wrap(Tensor& input_tensor);
}