#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;


void test_tensor_metadata_accessors() 
{
    cout << "\n=== test_tensor_metadata_accessors ===\n" << endl;

    cout << "Testing tensor metadata accessors...\n" << endl;
    
    // Test 1: Basic CPU tensor
    {
        Tensor t(Shape{{2, 3, 4}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        
        // Test shape accessor
        const auto& shape = t.shape().dims;
        assert(shape.size() == 3);
        assert(shape[0] == 2);
        assert(shape[1] == 3);
        assert(shape[2] == 4);
        cout << "✓ Shape accessor correct: [2, 3, 4]" << endl;
        
        // Test dtype accessor
        assert(t.dtype() == Dtype::Float32);
        cout << "✓ Dtype accessor correct: Float32" << endl;
        
        // Test device accessor
        assert(t.device().is_cpu());
        cout << "✓ Device accessor correct: CPU" << endl;
        
        // Test requires_grad accessor
        assert(t.requires_grad() == false);
        cout << "✓ Requires_grad accessor correct: false" << endl;
        
        // Test numel accessor
        assert(t.numel() == 24);
        cout << "✓ Numel accessor correct: 24" << endl;
        
        // Test ndim accessor
        assert(t.ndim() == 3);
        cout << "✓ Ndims accessor correct: 3" << endl;
    }
    
    // Test 2: GPU tensor with different properties
    {
        Tensor t(Shape{{5, 6}}, Dtype::Int64, DeviceIndex(Device::CUDA), true);
        
        // Test shape accessor
        const auto& shape = t.shape().dims;
        assert(shape.size() == 2);
        assert(shape[0] == 5);
        assert(shape[1] == 6);
        cout << "✓ Shape accessor correct: [5, 6]" << endl;
        
        // Test dtype accessor
        assert(t.dtype() == Dtype::Int64);
        cout << "✓ Dtype accessor correct: Int64" << endl;
        
        // Test device accessor
        assert(t.device().is_cuda());
        cout << "✓ Device accessor correct: CUDA" << endl;
        
        // Test requires_grad accessor
        assert(t.requires_grad() == true);
        cout << "✓ Requires_grad accessor correct: true" << endl;
        
        // Test numel accessor
        assert(t.numel() == 30);
        cout << "✓ Numel accessor correct: 30" << endl;
        
        // Test ndim accessor
        assert(t.ndim() == 2);
        cout << "✓ Ndims accessor correct: 2" << endl;
    }
    
    // Test 3: 1D tensor edge case
    {
        Tensor t(Shape{{100}}, Dtype::Float64, DeviceIndex(Device::CPU), true);
        
        const auto& shape = t.shape().dims;
        assert(shape.size() == 1);
        assert(shape[0] == 100);
        assert(t.dtype() == Dtype::Float64);
        assert(t.device().is_cpu());
        assert(t.requires_grad() == true);
        assert(t.numel() == 100);
        assert(t.ndim() == 1);
        
        cout << "✓ 1D tensor metadata correct: [100], Float64, CPU, requires_grad=true" << endl;
    }
    
    // Test 4: Using TensorOptions
    {
        TensorOptions opts;
        opts.dtype = Dtype::Int32;
        opts.device = DeviceIndex(Device::CUDA);
        opts.requires_grad = false;
        
        Tensor t(Shape{{3, 3, 3, 3}}, opts);
        
        const auto& shape = t.shape().dims;
        assert(shape.size() == 4);
        assert(shape[0] == 3);
        assert(shape[1] == 3);
        assert(shape[2] == 3);
        assert(shape[3] == 3);
        assert(t.dtype() == Dtype::Int32);
        assert(t.device().is_cuda());
        assert(t.requires_grad() == false);
        assert(t.numel() == 81);
        assert(t.ndim() == 4);
        
        cout << "✓ TensorOptions metadata correct: [3, 3, 3, 3], Int32, CUDA, requires_grad=false" << endl;
    }
    
    // Test 5: Zero-dimensional-like tensor (smallest non-empty)
    {
        Tensor t(Shape{{1}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        
        const auto& shape = t.shape().dims;
        assert(shape.size() == 1);
        assert(shape[0] == 1);
        assert(t.dtype() == Dtype::Float32);
        assert(t.device().is_cpu());
        assert(t.requires_grad() == false);
        assert(t.numel() == 1);
        assert(t.ndim() == 1);
        
        cout << "✓ 1-element tensor metadata correct: [1], Float32, CPU, requires_grad=false" << endl;
    }

    cout << "\nAll metadata accessor tests passed! ✓" << endl;
}

int main() 
{
    test_tensor_metadata_accessors();
}