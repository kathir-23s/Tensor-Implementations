#pragma once

#include "core/Tensor.h"

namespace OwnTensor{

// CPU wrappers - Out-of-place
Tensor square_out_cpu_wrap(const Tensor& input_tensor);
Tensor square_root_out_cpu_wrap(const Tensor& input_tensor);
Tensor reciprocal_out_cpu_wrap(const Tensor& input_tensor);
Tensor negator_out_cpu_wrap(const Tensor& input_tensor);
Tensor absolute_out_cpu_wrap(const Tensor& input_tensor);
Tensor sign_out_cpu_wrap(const Tensor& input_tensor);
// Power function - out-of-place (multiple overloads)
Tensor power_out_cpu_wrap(const Tensor& input_tensor, int exponent);
Tensor power_out_cpu_wrap(const Tensor& input_tensor, float exponent);
Tensor power_out_cpu_wrap(const Tensor& input_tensor, double exponent);

// CPU wrappers - In-place
void square_in_cpu_wrap(Tensor& input_tensor);
void square_root_in_cpu_wrap(Tensor& input_tensor);
void reciprocal_in_cpu_wrap(Tensor& input_tensor);
void negator_in_cpu_wrap(Tensor& input_tensor);
void absolute_in_cpu_wrap(Tensor& input_tensor);
void sign_in_cpu_wrap(Tensor& input_tensor);
// Power function - in-place (multiple overloads)
void power_in_cpu_wrap(Tensor& input_tensor, int exponent);
void power_in_cpu_wrap(Tensor& input_tensor, float exponent);
void power_in_cpu_wrap(Tensor& input_tensor, double exponent);

// GPU wrappers - Out-of-place
Tensor square_out_gpu_wrap(const Tensor& input_tensor);
Tensor square_root_out_gpu_wrap(const Tensor& input_tensor);
Tensor reciprocal_out_gpu_wrap(const Tensor& input_tensor);
Tensor negator_out_gpu_wrap(const Tensor& input_tensor);
Tensor absolute_out_gpu_wrap(const Tensor& input_tensor);
Tensor sign_out_gpu_wrap(const Tensor& input_tensor);
// Power function - out-of-place (multiple overloads)
Tensor power_out_gpu_wrap(const Tensor& input_tensor, int exponent);
Tensor power_out_gpu_wrap(const Tensor& input_tensor, float exponent);
Tensor power_out_gpu_wrap(const Tensor& input_tensor, double exponent);

// GPU wrappers - In-place
void square_in_gpu_wrap(Tensor& input_tensor);
void square_root_in_gpu_wrap(Tensor& input_tensor);
void reciprocal_in_gpu_wrap(Tensor& input_tensor);
void negator_in_gpu_wrap(Tensor& input_tensor);
void absolute_in_gpu_wrap(Tensor& input_tensor);
void sign_in_gpu_wrap(Tensor& input_tensor);
// Power function - in-place (multiple overloads)
void power_in_gpu_wrap(Tensor& input_tensor, int exponent);
void power_in_gpu_wrap(Tensor& input_tensor, float exponent);
void power_in_gpu_wrap(Tensor& input_tensor, double exponent);

} // namespace OwnTensor