#pragma once
#include "device/Allocator.h"

namespace OwnTensor
{
    class CUDAAllocator : public Allocator
    {
        public:
            void* allocate(size_t bytes) override;
            void deallocate(void* ptr) override;
            void memset(void* ptr, int value,size_t bytes) override;
            void memcpy(void* ptr, const void* src, size_t bytes) override;
    };
}