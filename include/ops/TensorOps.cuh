#pragma once
#include "core/Tensor.h"
#ifdef WITH_CUDA//✨✨✨
#include <driver_types.h>
#endif//✨✨✨

namespace OwnTensor
{
#ifdef WITH_CUDA

void cuda_add_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream = 0);//✨✨✨
void cuda_sub_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream = 0);//✨✨✨
void cuda_mul_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream = 0);//✨✨✨
void cuda_div_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream = 0);//✨✨✨


void cuda_add_tensor_inplace( Tensor& A, const Tensor& B, cudaStream_t stream = 0);//✨✨✨
void cuda_sub_tensor_inplace( Tensor& A, const Tensor& B, cudaStream_t stream = 0);//✨✨✨
void cuda_mul_tensor_inplace( Tensor& A, const Tensor& B, cudaStream_t stream = 0);//✨✨✨
void cuda_div_tensor_inplace( Tensor& A, const Tensor& B, cudaStream_t stream = 0);//✨✨✨

#endif
}//✨✨✨