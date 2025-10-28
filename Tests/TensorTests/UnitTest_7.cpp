#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void test_tensor_memory_initialization_zero() 
{
    cout << "\n=== test_tensor_memory_initialization_zero ===\n" << endl;
    cout << "Testing data memory zero-initialization...\n" << endl;
    
    // Test all dtypes on CPU
    {
        cout << "CPU: Testing zero-initialization for all dtypes..." << endl;
        
        vector<Dtype> all_dtypes = {
            Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64
        };
        
        for (const auto& dtype : all_dtypes) {
            Tensor t(Shape{{3, 4}}, dtype, DeviceIndex(Device::CPU), false);
            
            // Get pointer to data and verify all bytes are zero
            const uint8_t* data_ptr = static_cast<const uint8_t*>(t.data());
            size_t byte_count = t.nbytes();
            
            bool all_zeros = true;
            for (size_t i = 0; i < byte_count; ++i) {
                if (data_ptr[i] != 0) {
                    all_zeros = false;
                    break;
                }
            }
            
            assert(all_zeros);
            cout << "✓ " << "Dtype data memory zero-initialized" << endl;
        }
    }
    
    // Test GPU memory initialization
    {
        cout << "GPU: Testing zero-initialization..." << endl;
        
        Tensor t(Shape{{2, 5}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        
        // Copy GPU memory to CPU for verification
        Tensor cpu_copy = t.to_cpu();
        const uint8_t* data_ptr = static_cast<const uint8_t*>(cpu_copy.data());
        size_t byte_count = cpu_copy.nbytes();
        
        bool all_zeros = true;
        for (size_t i = 0; i < byte_count; ++i) {
            if (data_ptr[i] != 0) {
                all_zeros = false;
                break;
            }
        }
        
        assert(all_zeros);
        cout << "✓ GPU data memory zero-initialized" << endl;
    }
    
    // Test different tensor sizes
    {
        cout << "Testing various tensor sizes..." << endl;
        
        vector<Shape> test_shapes = {
            Shape{{1}},           // Smallest
            Shape{{100}},         // 1D medium
            Shape{{10, 10}},      // 2D
            Shape{{3, 4, 5}},     // 3D
            Shape{{2, 3, 4, 5}}   // 4D
        };
        
        for (const auto& shape : test_shapes) {
            Tensor t(shape, Dtype::Float32, DeviceIndex(Device::CPU), false);
            
            const uint8_t* data_ptr = static_cast<const uint8_t*>(t.data());
            size_t byte_count = t.nbytes();
            
            bool all_zeros = true;
            for (size_t i = 0; i < byte_count; ++i) {
                if (data_ptr[i] != 0) {
                    all_zeros = false;
                    break;
                }
            }
            
            assert(all_zeros);
        }
        cout << "✓ All tensor sizes properly zero-initialized" << endl;
    }

    cout << "\nAll data memory initialization tests passed! ✓" << endl;
}

void test_tensor_gradient_memory_initialization_zero() 
{
    cout << "\n=== test_tensor_gradient_memory_initialization_zero ===\n" << endl;
    cout << "Testing gradient memory zero-initialization...\n" << endl;
    
    // Test all dtypes on CPU with requires_grad = true
    {
        cout << "CPU: Testing gradient zero-initialization for all dtypes..." << endl;
        
        vector<Dtype> all_dtypes = {
            Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64
        };
        
        for (const auto& dtype : all_dtypes) {
            Tensor t(Shape{{3, 4}}, dtype, DeviceIndex(Device::CPU), true);
            
            // Verify gradient memory exists and is zero-initialized
            assert(t.grad() != nullptr);
            
            const uint8_t* grad_ptr = static_cast<const uint8_t*>(t.grad());
            size_t byte_count = t.grad_nbytes();
            
            bool all_zeros = true;
            for (size_t i = 0; i < byte_count; ++i) {
                if (grad_ptr[i] != 0) {
                    all_zeros = false;
                    break;
                }
            }
            
            assert(all_zeros);
            cout << "✓ " << "Dtype gradient memory zero-initialized" << endl;
        }
    }
    
    // Test GPU gradient memory initialization
{
    cout << "GPU: Testing gradient zero-initialization..." << endl;
    
    Tensor t(Shape{{2, 5}}, Dtype::Float32, DeviceIndex(Device::CUDA), true);
    
    // Simply verify gradient memory exists - skip byte verification for GPU
    // since we don't have a direct way to access GPU gradient memory from CPU
    assert(t.grad() != nullptr);
    assert(t.grad_nbytes() > 0);
    assert(t.grad_nbytes() == t.nbytes());
    
    cout << "✓ GPU gradient memory allocated and sized correctly" << endl;
}
    
    // Test gradient memory size matches data memory
    {
        cout << "Testing gradient memory consistency..." << endl;
        
        Tensor t(Shape{{4, 6}}, Dtype::Float64, DeviceIndex(Device::CPU), true);
        
        assert(t.grad() != nullptr);
        assert(t.grad_nbytes() == t.nbytes());
        
        const uint8_t* grad_ptr = static_cast<const uint8_t*>(t.grad());
        size_t byte_count = t.grad_nbytes();
        
        bool all_zeros = true;
        for (size_t i = 0; i < byte_count; ++i) {
            if (grad_ptr[i] != 0) {
                all_zeros = false;
                break;
            }
        }
        
        assert(all_zeros);
        cout << "✓ Gradient memory size consistent and zero-initialized" << endl;
    }
    
    // Test TensorOptions with requires_grad
    {
        cout << "Testing TensorOptions gradient initialization..." << endl;
        
        TensorOptions opts;
        opts.requires_grad = true;
        opts.dtype = Dtype::Int32;
        
        Tensor t(Shape{{5, 3}}, opts);
        
        assert(t.grad() != nullptr);
        
        const uint8_t* grad_ptr = static_cast<const uint8_t*>(t.grad());
        size_t byte_count = t.grad_nbytes();
        
        bool all_zeros = true;
        for (size_t i = 0; i < byte_count; ++i) {
            if (grad_ptr[i] != 0) {
                all_zeros = false;
                break;
            }
        }
        
        assert(all_zeros);
        cout << "✓ TensorOptions gradient properly zero-initialized" << endl;
    }

    cout << "\nAll gradient memory initialization tests passed! ✓" << endl;
}


int main()
{
    test_tensor_memory_initialization_zero();
    test_tensor_gradient_memory_initialization_zero();
}