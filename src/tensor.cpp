#include "tensor.h"

#include <iostream>


Tensor::Tensor(Shape shape, Dtype dtype, DeviceIndex device, bool requires_grad)
    : shape_(shape), dtype_(dtype), device_(device), requires_grad_(requires_grad) {
    
    // Calculate strides from shape dimensions
    
    // Validate shape has at least one dimension
    
    // Compute stride values in reverse order
    
    // Determine element size based on data type
    
    // Calculate total number of elements
    
    // Calculate total bytes needed
    
    // Allocate memory for tensor data based on device type
    
    // Handle CPU device allocation
    
    // Handle CUDA device allocation with device index
    
    // Allocate gradient memory if requires_grad is true
    
    // Zero initialize gradient memory if allocated
    
    // Set ownership flag
}