#pragma once
#include <cstddef>

namespace OwnTensor
{
    class Allocator 
    {
        public:
            virtual ~Allocator() = default;
            virtual void* allocate(size_t bytes) = 0;
            virtual void deallocate(void* ptr) = 0;
            virtual void memset(void* ptr, int value, size_t bytes) = 0;
            virtual void memcpy(void* dst, const void* src, size_t bytes) = 0;

    };
}