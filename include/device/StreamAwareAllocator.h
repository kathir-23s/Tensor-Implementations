#pragma once
#include "device/Allocator.h"
#include <cuda_runtime.h>

namespace OwnTensor
{
    class StreamAwareAllocator : public Allocator
    {
        public:
            virtual void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) = 0;
            virtual void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) = 0;
    };
}