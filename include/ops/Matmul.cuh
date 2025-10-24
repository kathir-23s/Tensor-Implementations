#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
    #ifdef WITH_CUDA
        void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output);
    #endif
}