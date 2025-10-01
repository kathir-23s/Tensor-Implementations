#include "../include/Tensor.h"
#include <iostream>
#include <cstring>


Tensor::Tensor(Shape shape, Dtype dtype, DeviceIndex device, bool requires_grad)
    : shape_(shape), dtype_(dtype), device_(device), requires_grad_(requires_grad) {
    
    // Validate shape has at least one dimension
    stride_.strides.resize(shape.dims.size());
    if (shape.dims.empty())
    {
        throw std::runtime_error("Shape must have atleast 1 Dimension");
    }

    // Calculate strides from shape dimensions
    stride_.strides[shape.dims.size()-1] = 1;
    for (int i = shape.dims.size() - 2; i >= 0; --i)
    {
        stride_.strides[i] = stride_.strides[i + 1] * shape.dims[i+1];
    }

        
    // Calculate total number of elements
    size_t total_elems = 1;
    for (auto dim : shape_.dims)
    {
        total_elems *= dim;
    }
    
    size_t elem_size = dtype_size(dtype);
    // Calculate total bytes needed
    size_t total_bytes = total_elems * elem_size;

    // Allocate memory for tensor data based on device type
    data_ptr_ = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes]);
    // Add zero-initialization for data too
    std::memset(data_ptr_.get(), 0, total_bytes);

    // Handle CPU device allocation
    
    // Handle CUDA device allocation with device index
    
    // Allocate gradient memory if requires_grad is true
    if (requires_grad_) {
        grad_ptr_ = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes]);
        
        // Zero initialize gradient buffer
        // Zero initialize gradient memory if allocated
        std::memset(grad_ptr_.get(), 0, total_bytes);
    }
    
    
    // Set ownership flag
    owns_data_ = true;
    if (requires_grad_) owns_grad_ = true;
    data_size_ = total_bytes;    
}

// Tensor Options constructor
Tensor::Tensor(Shape shape, TensorOptions opts)
    : Tensor(shape, opts.dtype, opts.device, opts.requires_grad) {
}


// Utility
size_t Tensor::numel() const 
{
    size_t total = 1;
    for (auto dim : shape_.dims) total *= dim;
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
    // How to do it 
    int64_t expected_stride = 1;
    for (int i = shape_.dims.size() - 1; i >= 0; --i)
    {
        if (stride_.strides[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape_.dims[i];
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