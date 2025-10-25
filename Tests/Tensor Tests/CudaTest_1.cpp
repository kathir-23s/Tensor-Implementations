#include "TensorLib.h"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cassert>

using namespace std;
using namespace OwnTensor;

void test_gpu_host_to_device_transfer() 
{
    cout << "\n=== test_gpu_host_to_device_transfer ===\n" << endl;
    cout << "Testing host-to-device data transfer...\n" << endl;
    
    // Test 1: Basic float data transfer
    {
        cout << "Testing float data transfer..." << endl;
        
        // Create source data on CPU
        vector<float> source_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        
        // Create GPU tensor
        Tensor gpu_tensor(Shape{{2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        
        // Transfer data from host to device
        gpu_tensor.set_data(source_data);
        
        // Transfer back to CPU for verification
        Tensor cpu_tensor = gpu_tensor.to_cpu();
        
        // Verify data matches
        const float* retrieved_data = static_cast<const float*>(cpu_tensor.data());
        bool data_matches = true;
        for (size_t i = 0; i < source_data.size(); ++i) {
            if (retrieved_data[i] != source_data[i]) {
                data_matches = false;
                cout << "Mismatch at index " << i << ": expected " << source_data[i] 
                     << ", got " << retrieved_data[i] << endl;
                break;
            }
        }
        
        assert(data_matches);
        cout << "✓ Float data transfer successful" << endl;
    }
    
    // Test 2: Integer data transfer
    {
        cout << "\nTesting integer data transfer..." << endl;
        
        vector<int32_t> source_data = {10, 20, 30, 40, 50, 60, 70, 80};
        
        Tensor gpu_tensor(Shape{{4, 2}}, Dtype::Int32, DeviceIndex(Device::CUDA), false);
        // cout << "Before set_data - device: " << (gpu_tensor.device().is_cuda() ? "CUDA" : "CPU") << endl;
        // cout << "Source data size: " << source_data.size() << endl;
        // cout << "Tensor numel: " << gpu_tensor.numel() << endl;
        // cout << "Dtype check: " << is_same_type<int32_t>(gpu_tensor.dtype()) << endl;
        gpu_tensor.set_data(source_data);
        
        Tensor cpu_tensor = gpu_tensor.to_cpu();
        // const int32_t* retrieved_data = static_cast<const int32_t*>(cpu_tensor.data());
        vector<int32_t> retrieved_data(cpu_tensor.numel());
        std::memcpy(retrieved_data.data(), cpu_tensor.data(), source_data.size() * sizeof(int32_t));
        
        bool data_matches = true;
        for (size_t i = 0; i < source_data.size(); ++i) {
            if (retrieved_data[i] != source_data[i]) {
                data_matches = false;
                break;
            }
        }
        
        assert(data_matches);
        cout << "✓ Integer data transfer successful" << endl;
    }
    
    // Test 3: Large data transfer
    {
        cout << "\nTesting large data transfer..." << endl;
        
        vector<float> source_data(1000);
        for (size_t i = 0; i < source_data.size(); ++i) {
            source_data[i] = static_cast<float>(i);
        }
        
        Tensor gpu_tensor(Shape{{100, 10}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        gpu_tensor.set_data(source_data);
        
        Tensor cpu_tensor = gpu_tensor.to_cpu();
        const float* retrieved_data = static_cast<const float*>(cpu_tensor.data());
        
        bool data_matches = true;
        for (size_t i = 0; i < source_data.size(); ++i) {
            if (retrieved_data[i] != source_data[i]) {
                data_matches = false;
                break;
            }
        }
        
        assert(data_matches);
        cout << "✓ Large data transfer successful" << endl;
    }
    
    // Test 4: Pattern verification
    {
        cout << "\nTesting pattern verification..." << endl;
        
        vector<double> source_data = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
        
        Tensor gpu_tensor(Shape{{3, 3}}, Dtype::Float64, DeviceIndex(Device::CUDA), false);
        gpu_tensor.set_data(source_data);
        
        Tensor cpu_tensor = gpu_tensor.to_cpu();
        const double* retrieved_data = static_cast<const double*>(cpu_tensor.data());
        
        // Verify specific pattern
        assert(retrieved_data[0] == 1.5);
        assert(retrieved_data[4] == 5.5); // center element
        assert(retrieved_data[8] == 9.5); // last element
        
        cout << "✓ Pattern verification successful" << endl;
    }

    cout << "\nAll host-to-device transfer tests passed!" << endl;
}

int main()
{
    test_gpu_host_to_device_transfer();
}