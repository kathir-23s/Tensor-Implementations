#pragma once
#include <cstddef>
//✨✨✨
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <cuda_runtime_api.h>

namespace OwnTensor
{
    class Allocator 
    {
        public:
            virtual ~Allocator() = default;
            virtual void* allocate(size_t bytes) = 0;
            virtual void deallocate(void* ptr) = 0;
            // virtual void memset(void* ptr, int value, size_t bytes) = 0;
            // virtual void memcpy(void* dst, const void* src, size_t bytes) = 0;

            // --- ASYNCHRONOUS (Stream-Aware) API ---
            // These are the core, high-performance functions.
            virtual void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) = 0;//✨✨✨
            virtual void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) = 0;//✨✨✨

            // --- SYNCHRONOUS (Convenience) API ---
            // These are simple wrappers for ease of use.
            virtual void memset(void* ptr, int value, size_t bytes) {
                #ifdef WITH_CUDA
                    memsetAsync(ptr, value, bytes, 0); // Use stream 0
                    cudaStreamSynchronize(0);          // Wait for stream 0 
                #else
                    // Fallback for non-CUDA build
                    (void)ptr; (void)value; (void)bytes; // Suppress warnings
                #endif
            }//✨✨✨

            virtual void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
                #ifdef WITH_CUDA
                    memcpyAsync(dst, src, bytes, kind, 0); // Use stream 0
                    cudaStreamSynchronize(0);            // Wait for stream 0
                #else
                    (void)dst; (void)src; (void)bytes; (void)kind; // Suppress warnings
                #endif
        }//✨✨✨
    };
}