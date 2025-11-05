#pragma once

#include "core/Tensor.h"
#ifdef WITH_CUDA //✨✨✨
    #include <driver_types.h>
#endif //✨✨✨

namespace OwnTensor {

// ============================================================================
// CPU Wrapper Declarations
// ============================================================================

// Basic trigonometric functions
Tensor sin_out_cpu_wrap(const Tensor& input);
void sin_in_cpu_wrap(Tensor& input);
Tensor cos_out_cpu_wrap(const Tensor& input);
void cos_in_cpu_wrap(Tensor& input);
Tensor tan_out_cpu_wrap(const Tensor& input);
void tan_in_cpu_wrap(Tensor& input);

// Inverse trigonometric functions
Tensor asin_out_cpu_wrap(const Tensor& input);
void asin_in_cpu_wrap(Tensor& input);
Tensor acos_out_cpu_wrap(const Tensor& input);
void acos_in_cpu_wrap(Tensor& input);
Tensor atan_out_cpu_wrap(const Tensor& input);
void atan_in_cpu_wrap(Tensor& input);

// Hyperbolic functions
Tensor sinh_out_cpu_wrap(const Tensor& input);
void sinh_in_cpu_wrap(Tensor& input);
Tensor cosh_out_cpu_wrap(const Tensor& input);
void cosh_in_cpu_wrap(Tensor& input);
Tensor tanh_out_cpu_wrap(const Tensor& input);
void tanh_in_cpu_wrap(Tensor& input);

// Inverse hyperbolic functions
Tensor asinh_out_cpu_wrap(const Tensor& input);
void asinh_in_cpu_wrap(Tensor& input);
Tensor acosh_out_cpu_wrap(const Tensor& input);
void acosh_in_cpu_wrap(Tensor& input);
Tensor atanh_out_cpu_wrap(const Tensor& input);
void atanh_in_cpu_wrap(Tensor& input);

// ============================================================================
// GPU Wrapper Declarations
// ============================================================================

// Basic trigonometric functions
Tensor sin_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void sin_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor cos_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void cos_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor tan_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void tan_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨

// Inverse trigonometric functions
Tensor asin_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void asin_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor acos_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void acos_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor atan_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void atan_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨

// Hyperbolic functions
Tensor sinh_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void sinh_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor cosh_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void cosh_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor tanh_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void tanh_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨

// Inverse hyperbolic functions
Tensor asinh_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void asinh_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor acosh_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void acosh_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨
Tensor atanh_out_gpu_wrap(const Tensor& input, cudaStream_t stream = 0);//✨✨✨
void atanh_in_gpu_wrap(Tensor& input, cudaStream_t stream = 0);//✨✨✨

} // namespace OwnTensor