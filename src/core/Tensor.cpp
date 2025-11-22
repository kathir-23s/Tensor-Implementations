#include "core/Tensor.h"
#include "dtype/Types.h"
#include "device/AllocatorRegistry.h"
#include "device/DeviceTransfer.h"
#include "device/Device.h"
#include "core/Views/ViewUtils.h"
#include <iostream>
#include <cstring>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include "core/Views/contiguous_kernel.h"
#include "ops/helpers/ConversionKernels.cuh"

#endif

namespace OwnTensor 
{
    Tensor::Tensor(Shape shape, Dtype dtype, DeviceIndex device)
        : shape_(shape), dtype_(dtype), device_(device){
        


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
            // throw std::runtime_error("Shape must have atleast 1 Dimension");
            return ;
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

        stride_ = ViewUtils::compute_strides(shape);
        storage_offset_ = 0;  // Initialize offset to 0
            
        // Calculate total number of elements
        size_t total_elems = numel();
        size_t elem_size = dtype_size(dtype);
        size_t raw_bytes = total_elems * elem_size;


        // Use raw bytes directly - no problematic alignment
        size_t total_bytes = raw_bytes;


        // size_t total_bytes;
        if (device.is_cpu())
        {
            total_bytes = (raw_bytes + 63) & ~63;
            
        }
        else 
        {
            total_bytes = ((raw_bytes + 256 - 1) / 256) * 256;

        }
        
        /*#######################################
                MEMORY ALLOCATION FOR DATA 
        #########################################*/

        // Handle CPU device allocation
        // Handle CUDA device allocation with device index
        Allocator* alloc = AllocatorRegistry::get_allocator(device.device);

        void* raw_data_ptr = alloc->allocate(total_bytes);

        #ifdef WITH_CUDA//✨✨✨
        if (device.is_cuda()) {
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            alloc->memsetAsync(raw_data_ptr, 0, total_bytes, stream);
        } else 
        #endif
        {
            alloc->memset(raw_data_ptr, 0, total_bytes);
        }//✨✨✨
        
        data_ptr_ = std::shared_ptr<uint8_t[]>(
            static_cast<uint8_t*>(raw_data_ptr),
            [alloc](uint8_t* ptr) { 
                alloc->deallocate(ptr); 
            }
        );

        // Set ownership flag
        owns_data_ = true;
        data_size_ = total_bytes;

    }

    // Tensor Options constructor
    Tensor::Tensor(Shape shape, TensorOptions opts)
        : Tensor(shape, opts.dtype, opts.device) {
    }

    // Private constructor for creating views (shares data pointer)
    Tensor::Tensor(std::shared_ptr<uint8_t[]> data_ptr,
                Shape shape, Stride stride, size_t offset, Dtype dtype, DeviceIndex device) :
                shape_(shape),
                stride_(stride),
                dtype_(dtype),
                device_(device),
                data_ptr_(data_ptr),
                owns_data_(false),
                storage_offset_(offset),
                data_size_(0)
    {
        // No memory allocation - sharing existing memory
    }

    
    // Utility
    size_t Tensor:: numel() const 
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
        return numel() * size_t(dtype_); // data_size_
    }


    size_t Tensor::storage_offset() const 
    {
        return storage_offset_;
    }

    // Determine element size based on data type
    size_t Tensor::dtype_size(Dtype d) {
        switch(d) {
            case Dtype::Bool: return 1;
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
        // Same device - just return this tensor (no copy needed)
        if (device.device == device_.device && device.index == device_.index)
        {
            return *this;
        }
        
        // Handle views: Must be contiguous before device transfer
        if (!owns_data_ || !is_contiguous())
        {
            throw std::runtime_error(
                "Non-contiguous tensors cannot be transferred. Implement contiguous() first."
            );
        }
        
        // Create tensor on target device
        Tensor result(shape_, dtype_, device);
        
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
    

    // bool 
    template const bool* Tensor::data<bool>() const;
    template bool* Tensor::data<bool>();

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
    // Assuming these types are correctly defined in dtype/Types.h
    template const float16_t* Tensor::data<float16_t>() const;
    template float16_t* Tensor::data<float16_t>();

    template const bfloat16_t* Tensor::data<bfloat16_t>() const;
    template bfloat16_t* Tensor::data<bfloat16_t>();

}