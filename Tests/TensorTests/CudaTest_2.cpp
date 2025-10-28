#include "TensorLib.h"
#include <cuda_runtime.h>
#include "device/DeviceTransfer.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;
using namespace OwnTensor;

void test_gpu_large_tensor_verification() 
{
    cout << "\n=== test_gpu_large_tensor_verification ===\n" << endl;
    cout << "Testing large tensor GPU memory allocation and verification..." << endl;
    
    // Create large tensor (adjust size based on your GPU memory)
    const size_t large_size = 100 * 1024 * 1024; // 100 million elements ~ 400MB for float
    cout << "Creating large tensor with " << large_size << " elements (~400MB for float)" << endl;
    cout << "Check htop and nvidia-smi now. Press Enter to continue...";
    
    
    Tensor gpu_tensor(Shape{{large_size}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
    cout << "GPU tensor created. Memory allocated on DEVICE (CUDA)." << endl;
    cout << "Tensor device: " << (gpu_tensor.device().is_cuda() ? "CUDA" : "CPU") << endl;
    cout << "nbytes(): " << gpu_tensor.nbytes() << " bytes" << endl;
    cout << "Check nvidia-smi for GPU memory usage. Press Enter to continue...";
    
    
    // Fill with negative sequence
    cout << "Preparing source data on HOST (CPU)..." << endl;
    vector<float> source_data(large_size);
    for (size_t i = 0; i < large_size; ++i) {
        source_data[i] = -static_cast<float>(i + 1); // -1, -2, -3, ...
    }
    
    cout << "Source data prepared on HOST (CPU). First 5 elements: ";
    for (int i = 0; i < 5; ++i) cout << source_data[i] << " ";
    cout << endl;
    cout << "Last 5 elements: ";
    for (int i = large_size - 5; i < large_size; ++i) cout << source_data[i] << " ";
    cout << endl;
    cout << "Press Enter to transfer from HOST (CPU) to DEVICE (GPU)...";
    
    
    // Transfer to GPU
    gpu_tensor.set_data(source_data);
    cout << "Data transferred from HOST (CPU) to DEVICE (GPU)." << endl;
    cout << "GPU tensor device: " << (gpu_tensor.device().is_cuda() ? "CUDA" : "CPU") << endl;
    cout << "Check nvidia-smi. Press Enter to verify data integrity on DEVICE (GPU)...";
    
    
    // Verify random elements are accessible from GPU
    cout << "Verifying random element access from DEVICE (GPU)..." << endl;
    vector<size_t> test_indices = {0, 1, large_size/2, large_size-2, large_size-1};
    for (size_t idx : test_indices) {
        // Create small tensor to verify single element
        Tensor test_element(Shape{{1}}, Dtype::Float32, DeviceIndex(Device::CPU), false);
        
        // Copy single element from DEVICE (GPU) to HOST (CPU) for verification
        device::copy_memory(test_element.data(), Device::CPU,
                          static_cast<const uint8_t*>(gpu_tensor.data()) + idx * sizeof(float), 
                          Device::CUDA,
                          sizeof(float));
        
        const float* retrieved = static_cast<const float*>(test_element.data());
        cout << "Element at index " << idx << " on DEVICE (GPU): expected " << source_data[idx] 
             << ", got " << retrieved[0] 
             << " | " << (retrieved[0] == source_data[idx] ? "✓ MATCH" : "✗ MISMATCH") << endl;
        assert(retrieved[0] == source_data[idx]);
    }
    
    // Full round-trip verification
    cout << "Performing full round-trip verification..." << endl;
    cout << "Press Enter to transfer entire tensor from DEVICE (GPU) back to HOST (CPU)...";
    
    
    Tensor cpu_roundtrip = gpu_tensor.to_cpu();
    cout << "Round-trip complete. Tensor now on HOST (CPU)." << endl;
    cout << "Round-trip tensor device: " << (cpu_roundtrip.device().is_cpu() ? "CPU" : "CUDA") << endl;
    cout << "Verifying all data on HOST (CPU)..." << endl;
    
    const float* final_data = static_cast<const float*>(cpu_roundtrip.data());
    bool all_match = true;
    
    // Check first, middle, and last elements
    vector<size_t> verify_indices = {0, large_size/4, large_size/2, 3*large_size/4, large_size-1};
    for (size_t idx : verify_indices) {
        if (final_data[idx] != source_data[idx]) {
            cout << "Mismatch at index " << idx << " on HOST (CPU): expected " << source_data[idx] 
                 << ", got " << final_data[idx] << endl;
            all_match = false;
            break;
        }
    }
    
    assert(all_match);
    cout << "✓ All verification points match on HOST (CPU)!" << endl;
    
    // Verify tensor metadata
    cout << "\nTensor metadata verification:" << endl;
    cout << "Original tensor location: " << (gpu_tensor.device().is_cuda() ? "DEVICE (GPU)" : "HOST (CPU)") << endl;
    cout << "Round-trip tensor location: " << (cpu_roundtrip.device().is_cpu() ? "HOST (CPU)" : "DEVICE (GPU)") << endl;
    cout << "Original data pointer: " << gpu_tensor.data() << " (points to GPU memory)" << endl;
    cout << "Round-trip data pointer: " << cpu_roundtrip.data() << " (points to CPU memory)" << endl;
    cout << "Pointers are " << (gpu_tensor.data() != cpu_roundtrip.data() ? "DIFFERENT" : "SAME") << " (expected: DIFFERENT - separate memory spaces)" << endl;
    
    cout << "\n✓ Large tensor GPU verification completed successfully!" << endl;
    cout << "Final check - Press Enter to clean up and exit test...";
    
}

int main()
{
    test_gpu_large_tensor_verification();
}