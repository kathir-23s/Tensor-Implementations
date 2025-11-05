#pragma once
#include "device/Allocator.h"

namespace OwnTensor
{
    class CPUAllocator : public Allocator {
    public:
        void* allocate(size_t bytes) override;
        void deallocate(void* ptr) override;
        void memset(void* ptr, int value, size_t bytes) override;
        void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) override;//✨✨✨
        
        // Asynchronous (matches base class)        
        void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override;//✨✨✨
        void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override;//✨✨✨

    };
}