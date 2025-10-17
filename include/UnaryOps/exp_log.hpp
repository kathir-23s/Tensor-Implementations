#pragma once
#include <cuda_runtime.h>
#include "../Tensor.h"

namespace OwnTensor{
    Tensor exp_out_cpu_wrap(const Tensor& input_tensor);
    void exp_in_cpu_wrap(Tensor& input_tensor);
    

    template<typename T, T(*func)(T)>
    __global__
    void unary_kernel_gpu(const T* in, T* out, size_t size);
    
    Tensor exp_out_gpu_wrap(const Tensor& input_tensor);
    void exp_in_gpu_wrap(Tensor& input_tensor);
}