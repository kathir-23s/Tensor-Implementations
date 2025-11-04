#include "device/CUDAAllocator.h"
#include <iostream>
#include <stdexcept>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{
    void* CUDAAllocator::allocate(size_t bytes) {
    #ifdef WITH_CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess || ptr == nullptr)
        {
            std::cerr << "CUDA Allocation Failed: " << cudaGetErrorString(err) 
                    << "(requested " << bytes / (1024*1024) << " MB" << std::endl;
            throw std::runtime_error("CUDA allcation failed");
        }
            // std::cout << "CUDAAllocator: Allocating " << bytes << " bytes (" 
            //   << bytes / (1024*1024) << " MB)" << std::endl;    
            return ptr;
    #else
        throw std::runtime_error("CUDA not available");
    #endif
    }

    // void CUDAAllocator::deallocate(void* ptr) {
    // #ifdef WITH_CUDA
    //     // cudaFree(ptr);
    //     if (ptr)
    //     {
    //         cudaError_t err = cudaFree(ptr);
    //         if (err != cudaSuccess) {
    //             std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
    //         }
    //     }
    // #endif
    // }

    void CUDAAllocator::deallocate(void* ptr) {
    #ifdef WITH_CUDA
        if (ptr) {
            // cudaDeviceSynchronize();//✨✨✨
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::string error_msg = std::string("CUDA free failed: ") + cudaGetErrorString(err);
                std::cerr << error_msg << std::endl;
                // DON'T throw here - might be called in destructor
                // But at least clear the error state
                cudaGetLastError();  // Clear error
            }
        }
    #endif
    }

    #ifdef WITH_CUDA//✨✨✨
    void CUDAAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {    
        cudaError_t err = cudaMemsetAsync(ptr, value, bytes, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(err));
        }

    }

    void CUDAAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
        }

    }
    #endif//✨✨✨

    void CUDAAllocator::memset(void* ptr, int value, size_t bytes) {
    #ifdef WITH_CUDA
        memsetAsync(ptr, value, bytes, 0);//✨✨✨
        cudaStreamSynchronize(0);
    #endif
    }

    void CUDAAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    #ifdef WITH_CUDA
        memcpyAsync(dst, src, bytes, kind, 0);//✨✨✨
        cudaStreamSynchronize(0);
    #endif
    }
}