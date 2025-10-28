#include "TensorLib.h"
#include <cuda_runtime.h>
#include "device/DeviceTransfer.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;
using namespace OwnTensor;

void test_gpu_data_persistence() 
{
    cout << "\n=== test_gpu_data_persistence ===\n" << endl;
    cout << "Testing data persistence across multiple GPU operations..." << endl;
    
    // Test 1: Multiple sequential transfers and accesses
    {
        cout << "Testing multiple sequential operations..." << endl;
        
        Tensor gpu_tensor(Shape{{6}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
        vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        
        // First transfer
        gpu_tensor.set_data(data1);
        cout << "First transfer completed" << endl;
        
        // Verify first transfer persisted
        Tensor verify1 = gpu_tensor.to_cpu();
        const float* check1 = static_cast<const float*>(verify1.data());
        assert(check1[0] == 1.0f && check1[5] == 6.0f);
        cout << "First verification passed" << endl;
        
        // Second transfer with different data
        vector<float> data2 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
        gpu_tensor.set_data(data2);
        cout << "Second transfer completed" << endl;
        
        // Verify second transfer persisted and replaced first
        Tensor verify2 = gpu_tensor.to_cpu();
        const float* check2 = static_cast<const float*>(verify2.data());
        assert(check2[0] == 10.0f && check2[5] == 60.0f);
        cout << "Second verification passed" << endl;
        
        // Third operation: partial modification via fill-like pattern
        vector<float> data3 = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f};
        gpu_tensor.set_data(data3);
        cout << "Third transfer completed" << endl;
        
        // Final verification
        Tensor verify3 = gpu_tensor.to_cpu();
        const float* check3 = static_cast<const float*>(verify3.data());
        assert(check3[0] == 100.0f && check3[5] == 600.0f);
        cout << "Third verification passed" << endl;
        
        cout << "✓ Multiple sequential operations persistence verified" << endl;
    }
    
    // Test 2: Persistence across multiple access patterns
    {
        cout << "Testing persistence across access patterns..." << endl;
        
        Tensor gpu_tensor(Shape{{8}}, Dtype::Int32, DeviceIndex(Device::CUDA), false);
        vector<int32_t> initial_data = {1, 2, 3, 4, 5, 6, 7, 8};
        gpu_tensor.set_data(initial_data);
        
        // Multiple random element accesses
        vector<size_t> access_indices = {0, 3, 5, 7};
        vector<int32_t> accessed_values;
        
        for (size_t idx : access_indices) {
            Tensor element_tensor(Shape{{1}}, Dtype::Int32, DeviceIndex(Device::CPU), false);
            device::copy_memory(element_tensor.data(), Device::CPU,
                              static_cast<const uint8_t*>(gpu_tensor.data()) + idx * sizeof(int32_t),
                              Device::CUDA, sizeof(int32_t));
            
            const int32_t* val = static_cast<const int32_t*>(element_tensor.data());
            accessed_values.push_back(val[0]);
        }
        
        // Verify all accessed values are correct and consistent
        assert(accessed_values[0] == 1);  // index 0
        assert(accessed_values[1] == 4);  // index 3  
        assert(accessed_values[2] == 6);  // index 5
        assert(accessed_values[3] == 8);  // index 7
        
        cout << "✓ Multiple access patterns persistence verified" << endl;
    }
    
    // Test 3: Data persistence after device-to-device transfers
    {
        cout << "Testing persistence after device transfers..." << endl;
        
        Tensor gpu_tensor1(Shape{{4}}, Dtype::Float64, DeviceIndex(Device::CUDA), false);
        vector<double> original_data = {1.1, 2.2, 3.3, 4.4};
        gpu_tensor1.set_data(original_data);
        
        // Transfer to another GPU tensor (device-to-device)
        Tensor gpu_tensor2 = gpu_tensor1.to_cuda();
        
        // Modify second tensor
        vector<double> modified_data = {5.5, 6.6, 7.7, 8.8};
        gpu_tensor2.set_data(modified_data);
        
        // Verify first tensor data persisted unchanged
        Tensor verify1 = gpu_tensor1.to_cpu();
        const double* check1 = static_cast<const double*>(verify1.data());
        assert(check1[0] == 1.1 && check1[3] == 4.4);
        
        // Verify second tensor has new data
        Tensor verify2 = gpu_tensor2.to_cpu();
        const double* check2 = static_cast<const double*>(verify2.data());
        assert(check2[0] == 5.5 && check2[3] == 8.8);
        
        cout << "✓ Device transfer persistence verified" << endl;
    }

    cout << "\nAll data persistence tests passed!" << endl;
}

void test_gpu_transfer_all_dtypes() 
{
    cout << "\n=== test_gpu_transfer_all_dtypes ===\n" << endl;
    cout << "Testing data transfers for ALL supported data types..." << endl;
    
    // Test all dtypes including the ones we haven't tested yet
    vector<pair<Dtype, string>> all_dtypes = {
        {Dtype::Int16, "Int16"},
        {Dtype::Int32, "Int32"},
        {Dtype::Int64, "Int64"},
        {Dtype::Float32, "Float32"},
        {Dtype::Float64, "Float64"}
        // Note: Float16 and Bfloat16 omitted as they need special handling
    };
    
    for (const auto& [dtype, dtype_name] : all_dtypes) {
        cout << "Testing " << dtype_name << " complete transfer cycle..." << endl;
        
        if (dtype == Dtype::Int16) {
            vector<int16_t> source = {100, -200, 300, -400};
            Tensor gpu_tensor(Shape{{4}}, dtype, DeviceIndex(Device::CUDA), false);
            gpu_tensor.set_data(source);
            
            Tensor cpu_tensor = gpu_tensor.to_cpu();
            const int16_t* result = static_cast<const int16_t*>(cpu_tensor.data());
            
            bool success = true;
            for (size_t i = 0; i < source.size(); ++i) {
                if (result[i] != source[i]) {
                    success = false;
                    break;
                }
            }
            assert(success);
            
        } else if (dtype == Dtype::Int32) {
            vector<int32_t> source = {1000, -2000, 3000, -4000, 5000};
            Tensor gpu_tensor(Shape{{5}}, dtype, DeviceIndex(Device::CUDA), false);
            gpu_tensor.set_data(source);
            
            Tensor cpu_tensor = gpu_tensor.to_cpu();
            const int32_t* result = static_cast<const int32_t*>(cpu_tensor.data());
            assert(memcmp(result, source.data(), source.size() * sizeof(int32_t)) == 0);
            
        } else if (dtype == Dtype::Int64) {
            vector<int64_t> source = {100000, -200000, 300000};
            Tensor gpu_tensor(Shape{{3}}, dtype, DeviceIndex(Device::CUDA), false);
            gpu_tensor.set_data(source);
            
            Tensor cpu_tensor = gpu_tensor.to_cpu();
            const int64_t* result = static_cast<const int64_t*>(cpu_tensor.data());
            assert(result[0] == 100000 && result[1] == -200000 && result[2] == 300000);
            
        } else if (dtype == Dtype::Float32) {
            vector<float> source = {1.5f, -2.5f, 3.5f, -4.5f};
            Tensor gpu_tensor(Shape{{4}}, dtype, DeviceIndex(Device::CUDA), false);
            gpu_tensor.set_data(source);
            
            Tensor cpu_tensor = gpu_tensor.to_cpu();
            const float* result = static_cast<const float*>(cpu_tensor.data());
            assert(result[0] == 1.5f && result[1] == -2.5f && result[2] == 3.5f && result[3] == -4.5f);
            
        } else if (dtype == Dtype::Float64) {
            vector<double> source = {1.25, -2.25, 3.25, -4.25, 5.25};
            Tensor gpu_tensor(Shape{{5}}, dtype, DeviceIndex(Device::CUDA), false);
            gpu_tensor.set_data(source);
            
            Tensor cpu_tensor = gpu_tensor.to_cpu();
            const double* result = static_cast<const double*>(cpu_tensor.data());
            assert(result[0] == 1.25 && result[1] == -2.25 && result[2] == 3.25 && result[3] == -4.25 && result[4] == 5.25);
        }
        
        cout << "✓ " << dtype_name << " full transfer cycle successful" << endl;
    }

    cout << "\nAll data type transfer tests passed!" << endl;
}

int main()
{
    test_gpu_data_persistence();
    test_gpu_transfer_all_dtypes();
}