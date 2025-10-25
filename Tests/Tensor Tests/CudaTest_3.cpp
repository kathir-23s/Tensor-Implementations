#include "TensorLib.h"
#include "device/DeviceTransfer.h"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cassert>

using namespace std;
// using OwnTensor::Device;
using namespace OwnTensor;

void test_gpu_partial_data_transfer() 
{
    cout << "\n=== test_gpu_partial_data_transfer ===\n" << endl;
    cout << "Testing partial data transfers between host and device..." << endl;
    
    // Test 1: Partial set_data with offset
    {
        cout << "Testing partial set_data with offset..." << endl;
        
        vector<float> source_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        Tensor gpu_tensor(Shape{{4, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        
        // Transfer only first 4 elements (first 2 rows)
        size_t partial_count = 4;
        size_t element_size = sizeof(float);
        
        cout << "Transferring " << partial_count << " elements out of " << source_data.size() << endl;
        device::copy_memory(gpu_tensor.data(), Device::CUDA,
                           source_data.data(), Device::CPU,
                           partial_count * element_size);
        
        // Verify partial transfer
        Tensor cpu_partial = gpu_tensor.to_cpu();
        const float* retrieved = static_cast<const float*>(cpu_partial.data());
        
        bool partial_match = true;
        for (size_t i = 0; i < partial_count; ++i) {
            if (retrieved[i] != source_data[i]) {
                cout << "Partial transfer mismatch at index " << i << endl;
                partial_match = false;
                break;
            }
        }
        
        assert(partial_match);
        cout << "✓ Partial set_data transfer successful" << endl;
    }
    
    // Test 2: Partial transfer with specific region
    {
        cout << "Testing partial transfer with specific region..." << endl;
        
        Tensor gpu_tensor(Shape{{10}}, Dtype::Int32, DeviceIndex(Device::CUDA), false);
        vector<int32_t> source_data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        
        // Transfer only elements 3-7 (zero-based indices)
        size_t start_idx = 3;
        size_t transfer_count = 5;
        
        cout << "Transferring elements " << start_idx << " to " << (start_idx + transfer_count - 1) << endl;
        
        device::copy_memory(static_cast<uint8_t*>(gpu_tensor.data()) + start_idx * sizeof(int32_t),
                           Device::CUDA,
                           source_data.data() + start_idx, Device::CPU,
                           transfer_count * sizeof(int32_t));
        
        // Verify specific region
        Tensor cpu_verify = gpu_tensor.to_cpu();
        const int32_t* retrieved = static_cast<const int32_t*>(cpu_verify.data());
        
        bool region_match = true;
        for (size_t i = start_idx; i < start_idx + transfer_count; ++i) {
            if (retrieved[i] != source_data[i]) {
                cout << "Region transfer mismatch at index " << i << endl;
                region_match = false;
                break;
            }
        }
        
        assert(region_match);
        cout << "✓ Partial region transfer successful" << endl;
    }
    
    // Test 3: Strided partial transfer (every other element)
    {
        cout << "Testing strided partial transfer..." << endl;
        
        Tensor gpu_tensor(Shape{{8}}, Dtype::Float64, DeviceIndex(Device::CUDA), false);
        vector<double> source_data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
        vector<double> expected_gpu = {1.1, 0.0, 3.3, 0.0, 5.5, 0.0, 7.7, 0.0};
        
        // Transfer every other element (stride of 2)
        for (size_t i = 0; i < source_data.size(); i += 2) {
            device::copy_memory(static_cast<uint8_t*>(gpu_tensor.data()) + i * sizeof(double),
                               Device::CUDA,
                               &source_data[i], Device::CPU,
                               sizeof(double));
        }
        
        // Verify strided pattern
        Tensor cpu_verify = gpu_tensor.to_cpu();
        const double* retrieved = static_cast<const double*>(cpu_verify.data());
        
        bool strided_match = true;
        for (size_t i = 0; i < source_data.size(); ++i) {
            double expected = (i % 2 == 0) ? source_data[i] : 0.0;
            if (retrieved[i] != expected) {
                cout << "Strided transfer mismatch at index " << i << endl;
                strided_match = false;
                break;
            }
        }
        
        assert(strided_match);
        cout << "✓ Strided partial transfer successful" << endl;
    }

    cout << "\nAll partial data transfer tests passed!" << endl;
}

void test_gpu_memory_isolation() 
{
    cout << "\n=== test_gpu_memory_isolation ===\n" << endl;
    cout << "Testing memory isolation between CPU and GPU..." << endl;
    
    // Test 1: Verify GPU modifications don't affect original CPU data
    {
        cout << "Testing GPU modifications don't affect CPU source..." << endl;
        
        vector<float> original_data = {1.0f, 2.0f, 3.0f, 4.0f};
        vector<float> cpu_backup = original_data; // Keep backup
        
        Tensor gpu_tensor(Shape{{4}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        gpu_tensor.set_data(original_data);
        
        // Modify the GPU data (simulate GPU operation by transferring modified data back)
        vector<float> modified_gpu_data = {10.0f, 20.0f, 30.0f, 40.0f};
        gpu_tensor.set_data(modified_gpu_data);
        
        // Verify original CPU data is unchanged
        bool cpu_unchanged = true;
        for (size_t i = 0; i < original_data.size(); ++i) {
            if (original_data[i] != cpu_backup[i]) {
                cout << "CPU data corrupted at index " << i << endl;
                cpu_unchanged = false;
                break;
            }
        }
        
        assert(cpu_unchanged);
        cout << "✓ CPU source data isolation verified" << endl;
    }
    
    // Test 2: Verify separate CPU and GPU tensors are independent
    {
        cout << "Testing independent CPU and GPU tensor memory..." << endl;
        
        // Create identical tensors on CPU and GPU
        vector<int32_t> initial_data = {100, 200, 300};
        Tensor cpu_tensor(Shape{{3}}, Dtype::Int32, DeviceIndex(Device::CPU), false);
        Tensor gpu_tensor(Shape{{3}}, Dtype::Int32, DeviceIndex(Device::CUDA), false);
        
        cpu_tensor.set_data(initial_data);
        gpu_tensor.set_data(initial_data);
        
        // Modify CPU tensor
        vector<int32_t> cpu_modification = {500, 600, 700};
        cpu_tensor.set_data(cpu_modification);
        
        // Verify GPU tensor is unchanged
        Tensor gpu_verify = gpu_tensor.to_cpu();
        const int32_t* gpu_data = static_cast<const int32_t*>(gpu_verify.data());
        
        bool gpu_unchanged = true;
        for (size_t i = 0; i < initial_data.size(); ++i) {
            if (gpu_data[i] != initial_data[i]) {
                cout << "GPU data affected by CPU modification at index " << i << endl;
                gpu_unchanged = false;
                break;
            }
        }
        
        assert(gpu_unchanged);
        cout << "✓ CPU/GPU tensor independence verified" << endl;
    }
    
    // Test 3: Verify transfers are explicit (no automatic sync)
    {
        cout << "Testing explicit transfer requirement..." << endl;
        
        Tensor gpu_tensor(Shape{{2}}, Dtype::Float64, DeviceIndex(Device::CUDA), false);
        vector<double> initial_data = {1.5, 2.5};
        gpu_tensor.set_data(initial_data);
        
        // Create CPU tensor but don't transfer yet
        Tensor cpu_tensor(Shape{{2}}, Dtype::Float64, DeviceIndex(Device::CPU), false);
        
        // Modify GPU data
        vector<double> gpu_modification = {9.9, 8.8};
        gpu_tensor.set_data(gpu_modification);
        
        // Now transfer - should get the modified data, not initial
        device::copy_memory(cpu_tensor.data(), Device::CPU,
                           gpu_tensor.data(), Device::CUDA,
                           2 * sizeof(double));
        
        const double* cpu_data = static_cast<const double*>(cpu_tensor.data());
        bool got_modified_data = (cpu_data[0] == 9.9 && cpu_data[1] == 8.8);
        
        assert(got_modified_data);
        cout << "✓ Explicit transfer requirement verified" << endl;
    }

    cout << "\nAll memory isolation tests passed!" << endl;
}



int main()
{
    test_gpu_partial_data_transfer();
    test_gpu_memory_isolation();
    // test_gpu_transfer_all_dtypes();
}