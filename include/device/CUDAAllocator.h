// include/device/CUDAAllocator.h

#pragma once
#include "device/StreamAwareAllocator.h"

namespace OwnTensor
{
    class CUDAAllocator : public StreamAwareAllocator
    {
        public:
            void* allocate(size_t bytes) override;
            void deallocate(void* ptr) override;
            void memset(void* ptr, int value, size_t bytes) override;
            void memcpy(void* ptr, const void* src, size_t bytes) override;

            //async
            void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override;//✨✨✨
            void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override;//✨✨✨

    };
}