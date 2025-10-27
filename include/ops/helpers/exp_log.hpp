#pragma once
#include "core/Tensor.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor{
// CPU wrappers - Out-of-place
Tensor exp_out_cpu_wrap(const Tensor& input_tensor);
Tensor exp2_out_cpu_wrap(const Tensor& input_tensor);
Tensor log_out_cpu_wrap(const Tensor& input_tensor);
Tensor log2_out_cpu_wrap(const Tensor& input_tensor);
Tensor log10_out_cpu_wrap(const Tensor& input_tensor);

// CPU wrappers - In-place
void exp_in_cpu_wrap(Tensor& input_tensor);
void exp2_in_cpu_wrap(Tensor& input_tensor);
void log_in_cpu_wrap(Tensor& input_tensor);
void log2_in_cpu_wrap(Tensor& input_tensor);
void log10_in_cpu_wrap(Tensor& input_tensor);

// GPU wrappers - Out-of-place
Tensor exp_out_gpu_wrap(const Tensor& input_tensor);
Tensor exp2_out_gpu_wrap(const Tensor& input_tensor);
Tensor log_out_gpu_wrap(const Tensor& input_tensor);
Tensor log2_out_gpu_wrap(const Tensor& input_tensor);
Tensor log10_out_gpu_wrap(const Tensor& input_tensor);

// GPU wrappers - In-place
void exp_in_gpu_wrap(Tensor& input_tensor);
void exp2_in_gpu_wrap(Tensor& input_tensor);
void log_in_gpu_wrap(Tensor& input_tensor);
void log2_in_gpu_wrap(Tensor& input_tensor);
void log10_in_gpu_wrap(Tensor& input_tensor);

}