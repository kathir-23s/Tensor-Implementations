#include "Tensor.h"
#include "Types.h"
#include "device/AllocatorRegistry.h"
#include "device/DeviceTransfer.h"
#include "device/Device.h"
#include <iostream>
#include <cstring>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#endif

#ifdef WITH_DEBUG
#endif

namespace OwnTensor 
{
    Tensor::Tensor(Shape shape, Dtype dtype, DeviceIndex device, bool requires_grad)
        : shape_(shape), dtype_(dtype), device_(device), requires_grad_(requires_grad) {
        
        #ifdef WITH_DEBUG
        std::cout << "\n=== TENSOR CONSTRUCTOR START ===" << std::endl;
        std::cout << "Tensor constructor: device=" << (device.is_cpu() ? "CPU" : "CUDA") << "\n" << std::endl;
        #endif

        // == CUDA DEVICE SETTING AND CHECK == //
        if (device.is_cuda())
        {
            #ifdef WITH_CUDA
        if (!device::cuda_available()) {
                throw std::runtime_error("CUDA is not available but CUDA device requested");
            }
            cudaError_t err = cudaSetDevice(device.index);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
            }
            // std::cout << "Set CUDA device to: " << device.index << std::endl;
        #else   
            throw std::runtime_error("CUDA support not compiled");        
            #endif
        }

        // Validate shape has at least one dimension
        stride_.strides.resize(shape.dims.size());

        if (shape.dims.empty())
        {
            throw std::runtime_error("Shape must have atleast 1 Dimension");
        }

        for (size_t i = 0; i < shape_.dims.size(); ++i) 
        {
            if (shape_.dims[i] < 0) 
            {
                throw std::runtime_error("All dimensions must be non-negative, got dimension " + 
                                        std::to_string(i) + " = " + std::to_string(shape_.dims[i]));
            }
            if (shape_.dims[i] == 0) 
            {        
                throw std::runtime_error("Zero dimensions are not allowed, got dimension " + 
                                        std::to_string(i) + " = 0");
            }
        }

        // Calculate strides from shape dimensions
        stride_.strides[shape.dims.size()-1] = 1;
        for (int i = shape.dims.size() - 2; i >= 0; --i)
        {
            stride_.strides[i] = stride_.strides[i + 1] * shape.dims[i+1];
        }

            
        // Calculate total number of elements
        size_t total_elems = numel();
        size_t elem_size = dtype_size(dtype);
        size_t raw_bytes = total_elems * elem_size;

        #ifdef WITH_DEBUG
        std::cout << "\n=== Memory calculation ===" << std::endl;
        std::cout << "  Elements: " << total_elems << std::endl;
        std::cout << "  Element size: " << elem_size << " bytes" << std::endl;
        std::cout << "  Raw bytes: " << raw_bytes << std::endl;
        std::cout << "  Raw MB: " << static_cast<double>(raw_bytes) / (1024 * 1024) << std::endl;
        #endif

        // Use raw bytes directly - no problematic alignment
        size_t total_bytes = raw_bytes;
        #ifdef WITH_DEBUG
        std::cout << "  Final allocation: " << total_bytes << " bytes (" 
                << static_cast<double>(total_bytes) / (1024 * 1024) << " MB)" << std::endl;
        #endif

        // size_t total_bytes;
        if (device.is_cpu())
        {
            total_bytes = (raw_bytes + 63) & ~63;
            #ifdef WITH_DEBUG
            std::cout << "  CPU Aligned bytes: " << total_bytes << std::endl;
            std::cout << "  Raw MB: " << static_cast<double>(total_bytes) / (1024 * 1024) << std::endl;
            #endif
        }
        else 
        {
            total_bytes = ((raw_bytes + 256 - 1) / 256) * 256;
            #ifdef WITH_DEBUG
            std::cout << "GPU Aligned bytes: " << total_bytes << std::endl;
            std::cout << "Raw MB: " << static_cast<double>(total_bytes) / (1024 * 1024) << std::endl;
            #endif

        }
        
        /*##############################################################
                MEMORY ALLOCATION FOR DATA AND GRADIENTS
        ################################################################*/

        // Handle CPU device allocation
        // Handle CUDA device allocation with device index
        Allocator* alloc = AllocatorRegistry::get_allocator(device.device);

        void* raw_data_ptr = alloc->allocate(total_bytes);
        if(!raw_data_ptr)
        {
            throw std::runtime_error("Data Memory Allocation Failed");
        } 

        alloc->memset(raw_data_ptr, 0, total_bytes);

        data_ptr_ = std::shared_ptr<uint8_t[]>(
            static_cast<uint8_t*>(raw_data_ptr),
            [alloc](uint8_t* ptr) { 
                alloc->deallocate(ptr); 
            }
        );

        if (requires_grad_) {
            void* raw_grad_ptr = alloc->allocate(total_bytes);
            if(!raw_grad_ptr) 
            {
                throw std::runtime_error("Gradient Memory Allocation Failed");
            }

            alloc->memset(raw_grad_ptr, 0, total_bytes);

            grad_ptr_ = std::shared_ptr<uint8_t[]>(
                static_cast<uint8_t*>(raw_grad_ptr),
                [alloc](uint8_t* ptr) { 
                    alloc->deallocate(ptr); 
                }
            );
        }
        
        // Set ownership flag
        owns_data_ = true;
        owns_grad_ = requires_grad_;
        data_size_ = total_bytes;

        // std::cout << "=== TENSOR CONSTRUCTOR END ===" << std::endl;    
    }

    // Tensor Options constructor
    Tensor::Tensor(Shape shape, TensorOptions opts)
        : Tensor(shape, opts.dtype, opts.device, opts.requires_grad) {
    }


    // Utility
    size_t Tensor::numel() const 
    {
        size_t total = 1;
        for (auto dim : shape_.dims) 
        {
        total *= dim;
        // std::cout << " numel: dim=" << dim << ", running_total=" << total << std::endl;
        }
        return total;
    }

    size_t Tensor::nbytes() const 
    {
        return data_size_;
    }

    size_t Tensor::grad_nbytes() const {
        if (requires_grad_){
            return data_size_;
        }
        else {
            return 0;
        }
    }

    bool Tensor::is_contiguous() const
    {
        // Need to look into it
        // What it is and what's it for
        int64_t expected_stride = 1;
        const auto& dims = shape_.dims;
        const auto& strides = stride_.strides;
       
        for (int i = dims.size() - 1; i >= 0; --i)
        {
            if (strides[i] != expected_stride)
            {
                return false;
            }
            expected_stride *= dims[i];
        }
        return true;
    }

    // Determine element size based on data type
    size_t Tensor::dtype_size(Dtype d) {
        switch(d) {
            case Dtype::Int16: return dtype_traits<Dtype::Int16>::size;
            case Dtype::Int32: return dtype_traits<Dtype::Int32>::size;
            case Dtype::Int64: return dtype_traits<Dtype::Int64>::size;
            case Dtype::Bfloat16: return dtype_traits<Dtype::Bfloat16>::size;
            case Dtype::Float16: return dtype_traits<Dtype::Float16>::size;
            case Dtype::Float32: return dtype_traits<Dtype::Float32>::size;
            case Dtype::Float64: return dtype_traits<Dtype::Float64>::size;
            default: throw std::runtime_error("Unsupported data type");
        }
    }

    Tensor Tensor::to(DeviceIndex device) const {
        if (device.device == device_.device && device.index == device_.index)
        {
            Tensor result(shape_, dtype_, device_, requires_grad_);
            Allocator* alloc = AllocatorRegistry::get_allocator(device_.device);
            alloc->memcpy(result.data(), data(), numel() * dtype_size(dtype_));
            return result;
        }
        
        // Create tensor on target device
        Tensor result(shape_, dtype_, device, requires_grad_);
        
        // Copy data between devices
        device::copy_memory(result.data(), device.device, 
                        data(), device_.device, 
                        numel() * dtype_size(dtype_));
        
        return result;
    }

    Tensor Tensor::to_cpu() const {
        return to(DeviceIndex(Device::CPU));
    }

    Tensor Tensor::to_cuda(int device_index) const {
        return to(DeviceIndex(Device::CUDA, device_index));
    }

    bool Tensor::is_cpu() const {
        return device_.is_cpu();
    }

    bool Tensor::is_cuda() const {
        return device_.is_cuda();
    }


    // int16_t (short)
    template const short* Tensor::data<short>() const;
    template short* Tensor::data<short>();

    // int32_t (int)
    template const int* Tensor::data<int>() const;
    template int* Tensor::data<int>();

    // int64_t (long/index type used for reduction output)
    template const int64_t* Tensor::data<int64_t>() const;
    template int64_t* Tensor::data<int64_t>(); 

    // float (float)
    template const float* Tensor::data<float>() const;
    template float* Tensor::data<float>();

    // double (double)
    template const double* Tensor::data<double>() const;
    template double* Tensor::data<double>();

    // Custom types (float16_t and bfloat16_t)
    // Assuming these types are correctly defined in types.h
    template const float16_t* Tensor::data<float16_t>() const;
    template float16_t* Tensor::data<float16_t>();

    template const bfloat16_t* Tensor::data<bfloat16_t>() const;
    template bfloat16_t* Tensor::data<bfloat16_t>();

}