#include "device/DeviceTransfer.h"
#include "device/AllocatorRegistry.h"
#include <stdexcept>
#include <iostream>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{
    namespace device {
        void copy_memory(void* dst, Device dst_device, 
                        const void* src, Device src_device, 
                        size_t bytes) {
            
            // std::cout << "DEBUG copy_memory: dst=" << dst 
            //           << " dst_device=" << (dst_device == Device::CPU ? "CPU" : "CUDA")
            //           << " src=" << src 
            //           << " src_device=" << (src_device == Device::CPU ? "CPU" : "CUDA")
            //           << " bytes=" << bytes << std::endl;
            
            // CPU to CPU
            if (dst_device == Device::CPU && src_device == Device::CPU) {
                // std::cout << "DEBUG: CPU->CPU transfer" << std::endl;
                Allocator* alloc = AllocatorRegistry::get_cpu_allocator();
                alloc->memcpy(dst, src, bytes);
                return;
            }
            
    #ifdef WITH_CUDA
            // GPU to GPU
            if (dst_device == Device::CUDA && src_device == Device::CUDA) {
                // std::cout << "DEBUG: GPU->GPU transfer" << std::endl;
                Allocator* alloc = AllocatorRegistry::get_cuda_allocator();
                alloc->memcpy(dst, src, bytes);
                return;
            }
            // CPU to GPU
            if (dst_device == Device::CUDA && src_device == Device::CPU) {
                // std::cout << "DEBUG: CPU->GPU transfer" << std::endl;
                cudaError_t result = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (result != cudaSuccess) {
                    std::cout << "DEBUG: CUDA error: " << cudaGetErrorString(result) << std::endl;
                }
                return;
            }
            // GPU to CPU  
            if (dst_device == Device::CPU && src_device == Device::CUDA) {
                // std::cout << "DEBUG: GPU->CPU transfer" << std::endl;
                cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                return;
            }
    #endif
            
            // If we get here, it's an unsupported transfer
            throw std::runtime_error("Unsupported device transfer");
        }
    }
}