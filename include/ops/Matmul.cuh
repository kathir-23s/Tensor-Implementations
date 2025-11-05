#pragma once

#include "core/Tensor.h"
#ifdef WITH_CUDA//✨✨✨
#include <driver_types.h>
#endif

namespace OwnTensor {
    #ifdef WITH_CUDA
        void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream = 0);//✨✨✨
    #endif
}