#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void test_tensor_gradient_memory_allocation() 
{
    cout << "\n=== test_tensor_gradient_memory_allocation ===\n" << endl;

    cout << "Testing gradient memory allocation...\n" << endl;
    
    // Test all dtypes with requires_grad = true (CPU)
    {
        cout << "CPU: Testing gradient allocation with requires_grad=true for all dtypes..." << endl;
        
        vector<Dtype> all_dtypes = {
            Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64
        };
        
        for (const auto& dtype : all_dtypes) {
            Tensor t(Shape{{4, 5}}, dtype, DeviceIndex(Device::CPU), true);
            
            // Gradient memory should be allocated
            assert(t.grad() != nullptr);
            assert(t.grad_nbytes() > 0);
            assert(t.owns_grad() == true);
            
            cout << "✓ " << "Dtype gradient allocated successfully" << endl;
        }
    }
    
    // Test all dtypes with requires_grad = false (CPU)
    {
        cout << "CPU: Testing no gradient allocation with requires_grad=false for all dtypes..." << endl;
        
        vector<Dtype> all_dtypes = {
            Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64
        };
        
        for (const auto& dtype : all_dtypes) {
            Tensor t(Shape{{4, 5}}, dtype, DeviceIndex(Device::CPU), false);
            
            // Gradient memory should not be allocated
            assert(t.grad() == nullptr);
            assert(t.grad_nbytes() == 0);
            assert(t.owns_grad() == false);
            
            cout << "✓ " << "Dtype gradient correctly not allocated" << endl;
        }
    }
    
    // Test GPU with requires_grad = true
    {
        cout << "GPU: Testing gradient allocation with requires_grad=true..." << endl;
        
        Tensor t(Shape{{3, 7}}, Dtype::Float32, DeviceIndex(Device::CUDA), true);
        
        // Gradient memory should be allocated
        assert(t.grad() != nullptr);
        assert(t.grad_nbytes() > 0);
        assert(t.owns_grad() == true);
        
        cout << "✓ GPU gradient allocated successfully" << endl;
    }
    
    // Test GPU with requires_grad = false
    {
        cout << "GPU: Testing no gradient allocation with requires_grad=false..." << endl;
        
        Tensor t(Shape{{3, 7}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        
        // Gradient memory should not be allocated
        assert(t.grad() == nullptr);
        assert(t.grad_nbytes() == 0);
        assert(t.owns_grad() == false);
        
        cout << "✓ GPU gradient correctly not allocated" << endl;
    }
    
    // Test gradient memory size matches data memory size
    {
        cout << "Testing gradient memory size matches data size..." << endl;
        
        Tensor t(Shape{{2, 8}}, Dtype::Float64, DeviceIndex(Device::CPU), true);
        
        assert(t.grad() != nullptr);
        assert(t.grad_nbytes() == t.nbytes());
        assert(t.owns_grad() == true);
        
        cout << "✓ Gradient memory size matches data size" << endl;
    }
    
    // Test TensorOptions with requires_grad
    {
        cout << "Testing TensorOptions requires_grad..." << endl;
        
        TensorOptions opts_true;
        opts_true.requires_grad = true;
        Tensor t1(Shape{{5, 2}}, opts_true);
        assert(t1.grad() != nullptr);
        
        TensorOptions opts_false;
        opts_false.requires_grad = false;
        Tensor t2(Shape{{5, 2}}, opts_false);
        assert(t2.grad() == nullptr);
        
        cout << "✓ TensorOptions requires_grad works correctly" << endl;
    }

    cout << "\nAll gradient memory allocation tests passed! ✓" << endl;
}

int main()
{
    test_tensor_gradient_memory_allocation();
}