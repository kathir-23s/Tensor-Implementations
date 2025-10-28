#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void test_tensor_numel_calculation() 
{
    cout << "\n=== test_tensor_numel_calculation ===\n" << endl;
    cout << "Testing tensor element count calculation...\n" << endl;
    
    // Test 1D tensors
    {
        cout << "Testing 1D tensors..." << endl;
        
        vector<pair<Shape, int64_t>> test_cases = {
            {Shape{{1}}, 1},
            {Shape{{10}}, 10},
            {Shape{{100}}, 100},
            {Shape{{1000}}, 1000}
        };
        
        for (const auto& [shape, expected] : test_cases) {
            Tensor t(shape, Dtype::Float32, DeviceIndex(Device::CPU), false);
            assert(t.numel() == expected);
            cout << "✓ 1D shape " << shape.dims[0] << ": numel=" << t.numel() << endl;
        }
    }
    
    // Test 2D tensors
    {
        cout << "Testing 2D tensors..." << endl;
        
        vector<pair<Shape, int64_t>> test_cases = {
            {Shape{{1, 1}}, 1},
            {Shape{{2, 3}}, 6},
            {Shape{{5, 5}}, 25},
            {Shape{{10, 20}}, 200},
            {Shape{{100, 50}}, 5000}
        };
        
        for (const auto& [shape, expected] : test_cases) {
            Tensor t(shape, Dtype::Float32, DeviceIndex(Device::CPU), false);
            assert(t.numel() == expected);
            cout << "✓ 2D shape [" << shape.dims[0] << ", " << shape.dims[1] << "]: numel=" << t.numel() << endl;
        }
    }
    
    // Test 3D tensors
    {
        cout << "Testing 3D tensors..." << endl;
        
        vector<pair<Shape, int64_t>> test_cases = {
            {Shape{{1, 1, 1}}, 1},
            {Shape{{2, 3, 4}}, 24},
            {Shape{{5, 5, 5}}, 125},
            {Shape{{3, 4, 5}}, 60},
            {Shape{{10, 10, 10}}, 1000}
        };
        
        for (const auto& [shape, expected] : test_cases) {
            Tensor t(shape, Dtype::Float32, DeviceIndex(Device::CPU), false);
            assert(t.numel() == expected);
            cout << "✓ 3D shape [" << shape.dims[0] << ", " << shape.dims[1] << ", " << shape.dims[2] << "]: numel=" << t.numel() << endl;
        }
    }
    
    // Test 4D tensors
    {
        cout << "Testing 4D tensors..." << endl;
        
        vector<pair<Shape, int64_t>> test_cases = {
            {Shape{{1, 1, 1, 1}}, 1},
            {Shape{{2, 2, 2, 2}}, 16},
            {Shape{{3, 4, 5, 6}}, 360},
            {Shape{{2, 3, 4, 5}}, 120}
        };
        
        for (const auto& [shape, expected] : test_cases) {
            Tensor t(shape, Dtype::Float32, DeviceIndex(Device::CPU), false);
            assert(t.numel() == expected);
            cout << "✓ 4D shape [" << shape.dims[0] << ", " << shape.dims[1] << ", " << shape.dims[2] << ", " << shape.dims[3] << "]: numel=" << t.numel() << endl;
        }
    }
    
    // Test edge cases
    {
        cout << "Testing edge cases..." << endl;
        
        // Single element tensor
        Tensor t1(Shape{{1}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        assert(t1.numel() == 1);
        
        // Large dimensions
        Tensor t2(Shape{{1000, 1000}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        assert(t2.numel() == 1000000);
        
        // Mixed dimensions
        Tensor t3(Shape{{2, 1, 5, 1, 3}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        assert(t3.numel() == 30);
        
        cout << "✓ All edge cases handled correctly" << endl;
    }
    
    // Test consistency across devices and data types
    {
        cout << "Testing consistency across devices and data types..." << endl;
        
        Shape test_shape{{3, 4, 5}};
        int64_t expected_numel = 60;
        
        // Test CPU with all data types
        vector<Dtype> all_dtypes = {
            Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, Dtype::Float64
        };
        
        for (const auto& dtype : all_dtypes) {
            Tensor t(test_shape, dtype, DeviceIndex(Device::CPU), false);
            assert(t.numel() == expected_numel);
        }
        
        // Test GPU
        Tensor t_gpu(test_shape, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        assert(t_gpu.numel() == expected_numel);
        
        cout << "✓ numel() consistent across all devices and data types" << endl;
    }
    
    // Test TensorOptions constructor
    {
        cout << "Testing TensorOptions numel calculation..." << endl;
        
        TensorOptions opts;
        Tensor t(Shape{{7, 8, 9}}, opts);
        assert(t.numel() == 504);
        
        cout << "✓ TensorOptions numel calculation correct" << endl;
    }

    cout << "\nAll numel calculation tests passed! ✓" << endl;
}

int main()
{
    test_tensor_numel_calculation();
}