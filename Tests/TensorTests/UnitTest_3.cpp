#include "TensorLib.h"
// #include "../include/Debug.h"

#include <iostream>
#include <cassert>
#include <vector>

using namespace OwnTensor;
using namespace std;

void test_tensor_default_constructor() 
{
    cout << "\n=== test_tensor_default_constructor ===\n" << endl;
    cout << "Testing default parameters...\n" << endl;
    
    // Test with shape and dtype
    {
        Tensor t2(Shape{{2, 3}}, Dtype::Float64);
        assert(t2.dtype() == Dtype::Float64);
        assert(t2.requires_grad() == false);
        cout << "✓ Shape + Dtype: dtype=Float64, requires_grad=false" << endl;
    }
    
    // Test with shape, dtype, and device (CPU)
    {
        Tensor t3(Shape{{2, 3}}, Dtype::Int32, DeviceIndex(Device::CPU));
        assert(t3.dtype() == Dtype::Int32);
        assert(t3.requires_grad() == false);
        cout << "✓ Shape + Dtype + Device(CPU): dtype=Int32, requires_grad=false" << endl;
    }
    
    // Test with shape, dtype, and device (GPU)
    {
        Tensor t4(Shape{{2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        assert(t4.dtype() == Dtype::Float32);
        assert(t4.requires_grad() == false);
        cout << "✓ Shape + Dtype + Device(GPU): dtype=Float32, requires_grad=false" << endl;
    }
    
    // Test with all parameters (CPU)
    {
        Tensor t5(Shape{{2, 3}}, Dtype::Float64, DeviceIndex(Device::CPU), true);
        assert(t5.dtype() == Dtype::Float64);
        assert(t5.requires_grad() == true);
        cout << "✓ All params (CPU): dtype=Float64, requires_grad=true" << endl;
    }
    
    // Test with all parameters (GPU)
    {
        Tensor t6(Shape{{2, 3}}, Dtype::Int64, DeviceIndex(Device::CUDA), true);
        assert(t6.dtype() == Dtype::Int64);
        assert(t6.requires_grad() == true);
        cout << "✓ All params (GPU): dtype=Int64, requires_grad=true" << endl;
    }
}

void test_tensor_options_constructor() 
{
    cout << "\n=== test_tensor_options_constructor ===\n" << endl;
    cout << "Testing TensorOptions constructor...\n" << endl;
    
    // Test default TensorOptions
    {
        TensorOptions opts;
        Tensor t1(Shape{{2, 3}}, opts);
        assert(t1.dtype() == Dtype::Float32);
        assert(t1.requires_grad() == false);
        cout << "✓ Default TensorOptions: dtype=Float32, requires_grad=false" << endl;
    }
    
    // Test with custom dtype
    {
        TensorOptions opts;
        opts.dtype = Dtype::Float64;
        Tensor t2(Shape{{2, 3}}, opts);
        assert(t2.dtype() == Dtype::Float64);
        cout << "✓ Custom dtype: dtype=Float64" << endl;
    }
    
    // Test with GPU device
    {
        TensorOptions opts;
        opts.device = DeviceIndex(Device::CUDA);
        Tensor t3(Shape{{2, 3}}, opts);
        assert(t3.dtype() == Dtype::Float32);
        cout << "✓ GPU device: device=CUDA" << endl;
    }
    
    // Test with requires_grad
    {
        TensorOptions opts;
        opts.requires_grad = true;
        Tensor t4(Shape{{2, 3}}, opts);
        assert(t4.requires_grad() == true);
        assert(t4.dtype() == Dtype::Float32);
        cout << "✓ Requires grad: requires_grad=true" << endl;
    }
    
    // Test builder pattern - with_dtype
    {
        TensorOptions opts;
        Tensor t5(Shape{{2, 3}}, opts.with_dtype(Dtype::Int32));
        assert(t5.dtype() == Dtype::Int32);
        cout << "✓ Builder pattern - with_dtype: dtype=Int32" << endl;
    }
    
    // Test builder pattern - with_device
    {
        TensorOptions opts;
        Tensor t6(Shape{{2, 3}}, opts.with_device(DeviceIndex(Device::CUDA)));
        assert(t6.dtype() == Dtype::Float32);
        cout << "✓ Builder pattern - with_device: device=CUDA" << endl;
    }
    
    // Test builder pattern - with_req_grad
    {
        TensorOptions opts;
        Tensor t7(Shape{{2, 3}}, opts.with_req_grad(true));
        assert(t7.requires_grad() == true);
        cout << "✓ Builder pattern - with_req_grad: requires_grad=true" << endl;
    }
    
    // Test combined builder pattern
    {
        TensorOptions opts;
        Tensor t8(Shape{{2, 3}}, opts.with_dtype(Dtype::Float64)
                                    .with_device(DeviceIndex(Device::CUDA))
                                    .with_req_grad(true));
        assert(t8.dtype() == Dtype::Float64);
        assert(t8.requires_grad() == true);
        cout << "✓ Combined builder: dtype=Float64, requires_grad=true" << endl;
    }
    
    // Test all custom options
    {
        TensorOptions opts;
        opts.dtype = Dtype::Int64;
        opts.device = DeviceIndex(Device::CUDA);
        opts.requires_grad = true;
        Tensor t9(Shape{{2, 3}}, opts);
        assert(t9.dtype() == Dtype::Int64);
        assert(t9.requires_grad() == true);
        cout << "✓ All custom options: dtype=Int64, requires_grad=true" << endl;
    }
}

int main() 
{
    test_tensor_default_constructor();
    test_tensor_options_constructor();
}