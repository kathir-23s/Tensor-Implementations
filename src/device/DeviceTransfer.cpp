#include "device/DeviceTransfer.h"
#include "device/AllocatorRegistry.h"
#include <stdexcept>
#include <iostream>
#include <driver_types.h>
#include "device/DeviceCore.h"//✨✨✨
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{
    namespace device {
        void copy_memory(void* dst, Device dst_device, 
                        const void* src, Device src_device, 
                        size_t bytes) {
            
            if (bytes == 0) {
                return; // Nothing to copy
            }
            
            // CPU to CPU
            if (dst_device == Device::CPU && src_device == Device::CPU) {
                Allocator* alloc = AllocatorRegistry::get_cpu_allocator();
                alloc->memcpy(dst, src, bytes, cudaMemcpyHostToDevice);//✨✨✨
                return;
            }
            
    #ifdef WITH_CUDA
            // GPU to GPU
            if (dst_device == Device::CUDA && src_device == Device::CUDA) {
                Allocator* alloc = AllocatorRegistry::get_cuda_allocator();
                alloc->memcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);//✨✨✨
                return;
            }
            // CPU to GPU
            if (dst_device == Device::CUDA && src_device == Device::CPU) {
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); // Get stream//✨✨✨
                cudaError_t result = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);//✨✨✨
                if (result != cudaSuccess) {
                    throw std::runtime_error(std::string("CPU->GPU transfer failed: ") + 
                                           cudaGetErrorString(result));
                }
                return;
            }
            // GPU to CPU  
            if (dst_device == Device::CPU && src_device == Device::CUDA) {
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); // Get stream//✨✨✨
                cudaError_t result = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);//✨✨✨
                if (result != cudaSuccess) {
                    throw std::runtime_error(std::string("GPU->CPU transfer failed: ") + 
                                           cudaGetErrorString(result));
                }
                return;
            }
    #endif
            
            // If we get here, it's an unsupported transfer
            throw std::runtime_error("Unsupported device transfer");
        }
    }
}