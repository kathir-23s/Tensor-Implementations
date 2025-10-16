#pragma once
#include <cuda_runtime.h>
#include "../Tensor.h"

namespace OwnTensor{
    Tensor exp_cpu_wrap(const Tensor& input_tensor);

    template<typename T, T(*func)(T)>
    __global__
    void unary_kernel_gpu(const T* in, T* out, size_t size);
    
    Tensor exp_gpu_wrap(const Tensor& input_tensor);


}