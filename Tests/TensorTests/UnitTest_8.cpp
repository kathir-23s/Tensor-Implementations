#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void test_tensor_ownership_flags() 
{
    cout << "\n=== test_tensor_ownership_flags ===\n" << endl;
    cout << "Testing tensor ownership flags...\n" << endl;
    
    // Test CPU tensor with requires_grad = false
    {
        cout << "Testing CPU tensor without gradients..." << endl;
        
        Tensor t(Shape{{3, 4}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        
        // Newly constructed tensor should own its data memory
        assert(t.owns_data() == true);
        
        // Without requires_grad, should not own gradient memory
        assert(t.owns_grad() == false);
        
        cout << "✓ CPU tensor ownership flags correct: owns_data=true, owns_grad=false" << endl;
    }
    
    // Test CPU tensor with requires_grad = true
    {
        cout << "Testing CPU tensor with gradients..." << endl;
        
        Tensor t(Shape{{3, 4}}, Dtype::Float32, DeviceIndex(Device::CPU), true);
        
        // Should own both data and gradient memory
        assert(t.owns_data() == true);
        assert(t.owns_grad() == true);
        
        cout << "✓ CPU tensor ownership flags correct: owns_data=true, owns_grad=true" << endl;
    }
    
    // Test GPU tensor with requires_grad = false
    {
        cout << "Testing GPU tensor without gradients..." << endl;
        
        Tensor t(Shape{{3, 4}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        
        // Should own data memory but not gradient memory
        assert(t.owns_data() == true);
        assert(t.owns_grad() == false);
        
        cout << "✓ GPU tensor ownership flags correct: owns_data=true, owns_grad=false" << endl;
    }
    
    // Test GPU tensor with requires_grad = true
    {
        cout << "Testing GPU tensor with gradients..." << endl;
        
        Tensor t(Shape{{3, 4}}, Dtype::Float32, DeviceIndex(Device::CUDA), true);
        
        // Should own both data and gradient memory
        assert(t.owns_data() == true);
        assert(t.owns_grad() == true);
        
        cout << "✓ GPU tensor ownership flags correct: owns_data=true, owns_grad=true" << endl;
    }
    
    // Test ownership flags across all data types
    {
        cout << "Testing ownership flags for all data types..." << endl;
        
        vector<Dtype> all_dtypes = {
            Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64
        };
        
        for (const auto& dtype : all_dtypes) {
            // Test without gradients
            Tensor t1(Shape{{2, 2}}, dtype, DeviceIndex(Device::CPU), false);
            assert(t1.owns_data() == true);
            assert(t1.owns_grad() == false);
            
            // Test with gradients
            Tensor t2(Shape{{2, 2}}, dtype, DeviceIndex(Device::CPU), true);
            assert(t2.owns_data() == true);
            assert(t2.owns_grad() == true);
        }
        
        cout << "✓ Ownership flags consistent across all data types" << endl;
    }
    
    // Test TensorOptions constructor
    {
        cout << "Testing TensorOptions ownership flags..." << endl;
        
        // Test with requires_grad = false
        TensorOptions opts_no_grad;
        opts_no_grad.requires_grad = false;
        Tensor t1(Shape{{4, 3}}, opts_no_grad);
        assert(t1.owns_data() == true);
        assert(t1.owns_grad() == false);
        
        // Test with requires_grad = true
        TensorOptions opts_with_grad;
        opts_with_grad.requires_grad = true;
        Tensor t2(Shape{{4, 3}}, opts_with_grad);
        assert(t2.owns_data() == true);
        assert(t2.owns_grad() == true);
        
        cout << "✓ TensorOptions ownership flags correct" << endl;
    }
    
    // Test various tensor shapes
    {
        cout << "Testing ownership flags for various shapes..." << endl;
        
        vector<Shape> test_shapes = {
            Shape{{1}},
            Shape{{100}},
            Shape{{5, 5}},
            Shape{{2, 3, 4}},
            Shape{{2, 2, 2, 2}}
        };
        
        for (const auto& shape : test_shapes) {
            Tensor t(shape, Dtype::Float32, DeviceIndex(Device::CPU), true);
            assert(t.owns_data() == true);
            assert(t.owns_grad() == true);
        }
        
        cout << "✓ Ownership flags consistent across all tensor shapes" << endl;
    }

    cout << "\nAll ownership flag tests passed! ✓" << endl;
}

int main()
{
    test_tensor_ownership_flags();
}