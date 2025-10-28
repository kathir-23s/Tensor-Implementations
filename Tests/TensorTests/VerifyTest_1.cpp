#include "TensorLib.h"
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace OwnTensor;

void test_cuda_connection_verification() {
    cout << "=========================================" << endl;
    cout << "    CUDA CONNECTION VERIFICATION TEST" << endl;
    cout << "=========================================" << endl;
    
    // Test 1: Check if CUDA is available
    cout << "\n1. Checking CUDA availability..." << endl;
    try {
        Tensor test_tensor(Shape{{1}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        cout << "✅ CUDA is available and working!" << endl;
        cout << "   Tensor device: " << (test_tensor.device().is_cuda() ? "CUDA" : "CPU") << endl;
    } catch (const std::exception& e) {
        cout << "❌ CUDA not available: " << e.what() << endl;
        return;
    }

    cout << "\n2. Check nvidia-smi NOW - you should see GPU memory usage." << endl;
    cout << "   Press Enter to continue...";
    cin.get();

    // Test 2: Allocate significant GPU memory
    cout << "\n3. Allocating large GPU tensor (100MB)..." << endl;
    try {
        const size_t large_size = 25 * 1024 * 1024; // 25M elements ~ 100MB for float
        Tensor large_tensor(Shape{{large_size}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        
        cout << "✅ Large GPU allocation successful!" << endl;
        cout << "   Tensor size: " << large_tensor.numel() << " elements" << endl;
        cout << "   Memory: " << large_tensor.nbytes() / (1024*1024) << " MB" << endl;
        
        cout << "\n4. Check nvidia-smi again - memory usage should increase." << endl;
        cout << "   Press Enter to continue...";
        cin.get();
        
    } catch (const std::exception& e) {
        cout << "❌ Large GPU allocation failed: " << e.what() << endl;
    }

    // Test 3: GPU memory operations
    cout << "\n5. Testing GPU memory operations..." << endl;
    try {
        Tensor gpu_tensor(Shape{{1000}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        vector<float> source_data(1000, 42.0f);
        
        gpu_tensor.set_data(source_data);
        cout << "✅ GPU memory write successful!" << endl;
        
        Tensor cpu_tensor = gpu_tensor.to_cpu();
        const float* retrieved = static_cast<const float*>(cpu_tensor.data());
        
        if (retrieved[0] == 42.0f && retrieved[999] == 42.0f) {
            cout << "✅ GPU memory read successful!" << endl;
        } else {
            cout << "❌ GPU memory verification failed!" << endl;
        }
        
    } catch (const std::exception& e) {
        cout << "❌ GPU memory operations failed: " << e.what() << endl;
    }

    // Test 4: Multiple GPU tensors
    cout << "\n6. Creating multiple GPU tensors..." << endl;
    try {
        vector<unique_ptr<Tensor>> gpu_tensors;
        for (int i = 0; i < 5; ++i) {
            gpu_tensors.push_back(make_unique<Tensor>(
                Shape{{1024 * 1024}}, Dtype::Float32, DeviceIndex(Device::CUDA), false
            ));
        }
        cout << "✅ Multiple GPU tensors created successfully!" << endl;
        cout << "   Total allocated: ~20MB (5 x 4MB)" << endl;
        
    } catch (const std::exception& e) {
        cout << "❌ Multiple GPU tensors failed: " << e.what() << endl;
    }

    cout << "\n7. Final nvidia-smi check - all memory should be properly managed." << endl;
    cout << "   Press Enter to exit...";
    cin.get();

    cout << "\n=========================================" << endl;
    cout << "     CUDA VERIFICATION COMPLETE!" << endl;
    cout << "=========================================" << endl;
}

int main() {
    test_cuda_connection_verification();
    return 0;
}