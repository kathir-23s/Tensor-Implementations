#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void tensor_cpu_memory_allocation_size() 
{
    cout << "=== Memory Allocation Size Test (CPU) ===\n" << endl;

    cout << "=== Smaller Memory Size Test for all Combinations ===\n" << endl;
    {
    Tensor ta(Shape{{2, 3}}, Dtype::Int16, DeviceIndex(Device::CPU, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 64 for CPU)
    assert(allocated_bytes % 64 == 0);
    
    cout << "✓ CPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 64 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{5, 3}}, Dtype::Int32, DeviceIndex(Device::CPU, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 64 for CPU)
    assert(allocated_bytes % 64 == 0);
    
    cout << "✓ CPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 64 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{8, 3}}, Dtype::Int64, DeviceIndex(Device::CPU, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 64 for CPU)
    assert(allocated_bytes % 64 == 0);
    
    cout << "✓ CPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 64 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{7, 4}}, Dtype::Float32, DeviceIndex(Device::CPU, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 64 for CPU)
    assert(allocated_bytes % 64 == 0);
    
    cout << "✓ CPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 64 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{6, 4}}, Dtype::Float64, DeviceIndex(Device::CPU, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 64 for CPU)
    assert(allocated_bytes % 64 == 0);
    
    cout << "✓ CPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 64 == 0 ? "valid" : "invalid") << endl;
    }

}

void tensor_cuda_memory_allocation_size() 
{
    cout << "=== Memory Allocation Size Test (CUDA) ===\n" << endl;

    cout << "=== Smaller Memory Size Test for all Combinations ===\n" << endl;
    {
    Tensor ta(Shape{{2, 3}}, Dtype::Int16, DeviceIndex(Device::CUDA, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 256 for GPU)
    assert(allocated_bytes % 256 == 0);
    
    cout << "✓ GPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 256 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{5, 3}}, Dtype::Int32, DeviceIndex(Device::CUDA, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 256 for GPU)
    assert(allocated_bytes % 256 == 0);
    
    cout << "✓ GPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 256 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{8, 3}}, Dtype::Int64, DeviceIndex(Device::CUDA, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 256 for GPU)
    assert(allocated_bytes % 256 == 0);
    
    cout << "✓ GPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 256 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{7, 4}}, Dtype::Float32, DeviceIndex(Device::CUDA, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 256 for GPU)
    assert(allocated_bytes % 256 == 0);
    
    cout << "✓ GPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 256 == 0 ? "valid" : "invalid") << endl;
    }

    {
    Tensor ta(Shape{{6, 4}}, Dtype::Float64, DeviceIndex(Device::CUDA, 0));
    Dtype dtype = ta.dtype();
    size_t required_bytes = ta.numel() * Tensor::dtype_size(dtype);
    size_t allocated_bytes = ta.nbytes();
    
    // Check that allocated bytes meet or exceed required bytes
    assert(allocated_bytes >= required_bytes);
    
    // Check that allocation is properly aligned (multiple of 256 for GPU)
    assert(allocated_bytes % 256 == 0);
    
    cout << "✓ GPU allocation correct: required=" << required_bytes 
         << ", allocated=" << allocated_bytes 
         << ", alignment=" << (allocated_bytes % 256 == 0 ? "valid" : "invalid") << endl;
    }

}

int main() 
{
    tensor_cpu_memory_allocation_size();
    tensor_cuda_memory_allocation_size();
}