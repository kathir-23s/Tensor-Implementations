

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

    // Asynchronous versions for CPU just call the standard library functions (and ignore the stream)
    void CPUAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t /*stream*/) //✨✨✨
    {
        std::memset(ptr, value, bytes);
    }

    void CPUAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, [[maybe_unused]]cudaMemcpyKind /*kind*/, [[maybe_unused]] cudaStream_t /*stream*/) //✨✨✨
    {
        std::memcpy(dst, src, bytes);
    }

    // Synchronous versions can simply call the async versions (which are synchronous on CPU anyway)
    void CPUAllocator::memset(void* ptr, int value, size_t bytes)//✨✨✨
    {
        std::memset(ptr, value, bytes);
    }

    void CPUAllocator::memcpy(void* dst, const void* src, size_t bytes, [[maybe_unused]] cudaMemcpyKind kind) //✨✨✨
    {
        std::memcpy(dst, src, bytes);
    }

}