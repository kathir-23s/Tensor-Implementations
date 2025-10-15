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

    void CUDAAllocator::deallocate(void* ptr) {
    #ifdef WITH_CUDA
        // cudaFree(ptr);
        if (ptr)
        {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    #endif
    }

    void CUDAAllocator::memset(void* ptr, int value, size_t bytes) {
    #ifdef WITH_CUDA
        // cudaMemset(ptr, value, bytes);
        cudaError_t err = cudaMemset(ptr, value, bytes);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaMemset failed: ") + cudaGetErrorString(err));
        }
    #endif
    }

    void CUDAAllocator::memcpy(void* dst, const void* src, size_t bytes) {
    #ifdef WITH_CUDA
        // cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
        }
    #endif
    }
}