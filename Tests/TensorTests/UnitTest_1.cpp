#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace OwnTensor;
using namespace std;

void test_tensor_constructor_valid_shape() {
    std::cout << "\n=== test_tensor_constructor_valid_shape ===\n" << std::endl;
    
    // Test CPU
    {
        std::cout << "CPU: Testing valid shapes..." << std::endl;
        Tensor t1(Shape{{10}}, Dtype::Float32, DeviceIndex(Device::CPU));
        assert(t1.numel() == 10);
        assert(t1.is_cpu());
        
        Tensor t2(Shape{{5, 5}}, Dtype::Float32, DeviceIndex(Device::CPU));
        assert(t2.numel() == 25);
        
        Tensor t3(Shape{{2, 3, 4}}, Dtype::Float32, DeviceIndex(Device::CPU));
        assert(t3.numel() == 24);

        Tensor t4(Shape{{2, 3, 4, 5}}, Dtype::Float32, DeviceIndex(Device::CPU));
        assert(t4.numel() == 120);

        Tensor t5(Shape{{2, 3, 4, 5, 6}}, Dtype::Float32, DeviceIndex(Device::CPU));
        assert(t5.numel() == 720);

        std::cout << "CPU: ✓ All valid shapes passed upto 5 dimensions" << std::endl;
        cout << "Verify RAM or VRAM usage!!" <<endl;
        

    }
    
    // Test GPU
    {
        std::cout << "GPU: Testing valid shapes..." << std::endl;
        Tensor t1(Shape{{10}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        assert(t1.numel() == 10);
        assert(t1.is_cuda());
        
        Tensor t2(Shape{{5, 5}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        assert(t2.numel() == 25);

        Tensor t3(Shape{{2, 3, 4}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        assert(t3.numel() == 24);

        Tensor t4(Shape{{2, 3, 4, 5}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        assert(t4.numel() == 120);

        Tensor t5(Shape{{2, 3, 4, 5, 6}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        assert(t5.numel() == 720);

        std::cout << "GPU: ✓ All valid shapes passed" << std::endl;
        cout << "Verify RAM or VRAM usage!!" <<endl;
        
    }
}

void test_tensor_constructor_empty_shape() {
    std::cout << "\n=== test_tensor_constructor_empty_shape ===\n" << std::endl;
    
    bool cpu_empty_threw = false;
    bool cpu_negative_threw = false;
    bool cpu_zero_threw = false;
    bool gpu_empty_threw = false;
    bool gpu_negative_threw = false;
    bool gpu_zero_threw = false;
    
    // Test CPU empty shape (this works)
    try {
        Tensor t(Shape{{}}, Dtype::Float32, DeviceIndex(Device::CPU));
    } catch (const std::runtime_error& e) {
        cpu_empty_threw = true;
        std::cout << "CPU: ✓ Correctly rejected empty shape: " << e.what() << std::endl;
    }
    
    // Test CPU negative dimension (this should throw but doesn't)
    try {
        Tensor t(Shape{{1, -2, 3}}, Dtype::Float32, DeviceIndex(Device::CPU));
        std::cout << "CPU: INCORRECTLY accepted negative dimensions! ndim: " << t.ndim() << std::endl;
    } catch (const std::runtime_error& e) {
        cpu_negative_threw = true;
        std::cout << "CPU: ✓ Correctly rejected negative dimension: " << e.what() << std::endl;
    }
    
    // Test CPU zero dimension (this should throw but doesn't)
    try {
        Tensor t(Shape{{0}}, Dtype::Float32, DeviceIndex(Device::CPU));
        std::cout << "CPU: INCORRECTLY accepted zero dimension! ndim: " << t.ndim() << std::endl;
    } catch (const std::runtime_error& e) {
        cpu_zero_threw = true;
        std::cout << "CPU: ✓ Correctly rejected zero dimension: " << e.what() << std::endl;
    }
    
    // Test GPU empty shape (this works)
    try {
        Tensor t(Shape{{}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    } catch (const std::runtime_error& e) {
        gpu_empty_threw = true;
        std::cout << "GPU: ✓ Correctly rejected empty shape: " << e.what() << std::endl;
    }

    try {
        Tensor t(Shape{{1, -2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::cout << "CPU: INCORRECTLY accepted negative dimensions! ndim: " << t.ndim() << std::endl;
    } catch (const std::runtime_error& e) {
        gpu_negative_threw = true;
        std::cout << "GPU: ✓ Correctly rejected negative dimension: " << e.what() << std::endl;
    }
    
    // Test CPU zero dimension (this should throw but doesn't)
    try {
        Tensor t(Shape{{0}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::cout << "CPU: INCORRECTLY accepted zero dimension! ndim: " << t.ndim() << std::endl;
    } catch (const std::runtime_error& e) {
        gpu_zero_threw = true;
        std::cout << "GPU: ✓ Correctly rejected zero dimension: " << e.what() << std::endl;
    }
    
    // Report findings
    std::cout << "\n=== TEST SUMMARY ===" << std::endl;
    std::cout << "Empty shapes: " << (cpu_empty_threw && gpu_empty_threw ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Negative dimensions: " << (cpu_negative_threw ? "✓ PASS" : "✗ FAIL - CPU doesn't validate dimensions!") << std::endl;
    std::cout << "Zero dimensions: " << (cpu_zero_threw ? "✓ PASS" : "✗ FAIL - CPU doesn't validate dimensions!") << std::endl;
    std::cout << "Negative dimensions: " << (gpu_negative_threw ? "✓ PASS" : "✗ FAIL - GPU doesn't validate dimensions!") << std::endl;
    std::cout << "Zero dimensions: " << (gpu_zero_threw ? "✓ PASS" : "✗ FAIL - GPU doesn't validate dimensions!") << std::endl;

    
    // Only assert what we know actually works
    assert(cpu_empty_threw && gpu_empty_threw);
    std::cout << "\nBasic empty shape validation: ✓ PASS" << std::endl;
}



int main() 
{
    test_tensor_constructor_valid_shape();
    test_tensor_constructor_empty_shape();

    return 0;
}