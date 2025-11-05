#pragma once

#include "core/Tensor.h"
#ifdef WITH_CUDA//✨✨✨
#include <driver_types.h>
#endif

namespace OwnTensor
{
    Tensor matmul(const Tensor& A, const Tensor& B, cudaStream_t stream = 0);//✨✨✨
}



