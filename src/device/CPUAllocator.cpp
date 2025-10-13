

#include "device/CPUAllocator.h"
#include <cstdlib>
#include <cstring>
#include <memory>

namespace OwnTensor
{
    void* CPUAllocator::allocate(size_t bytes)
    {
        return new uint8_t[bytes];
    }

    void CPUAllocator::deallocate(void* ptr)
    {
        delete[] static_cast<uint8_t*>(ptr);
    }

    void CPUAllocator::memset(void* ptr, int value, size_t bytes)
    {
        std::memset(ptr, value, bytes);
    }

    void CPUAllocator::memcpy(void* dst, const void* src, size_t bytes)
    {
        std::memcpy(dst, src, bytes);
    }

}