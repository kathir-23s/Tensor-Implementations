#pragma once
#include "core/Tensor.h"

namespace OwnTensor
{
#ifdef WITH_CUDA

void cuda_add_tensor(const Tensor& A, const Tensor& B, Tensor& output);
void cuda_sub_tensor(const Tensor& A, const Tensor& B, Tensor& output);
void cuda_mul_tensor(const Tensor& A, const Tensor& B, Tensor& output);
void cuda_div_tensor(const Tensor& A, const Tensor& B, Tensor& output);


void cuda_add_tensor_inplace( Tensor& A, const Tensor& B);
void cuda_sub_tensor_inplace( Tensor& A, const Tensor& B);
void cuda_mul_tensor_inplace( Tensor& A, const Tensor& B);
void cuda_div_tensor_inplace( Tensor& A, const Tensor& B);

#endif
}