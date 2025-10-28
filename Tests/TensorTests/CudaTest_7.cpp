#include "TensorLib.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace std;
using namespace OwnTensor;

void test_gpu_factory_consistency() 
{
    cout << "\n=== test_gpu_factory_consistency ===\n" << endl;
    cout << "Testing consistency between GPU and CPU factory functions..." << endl;
    
    // Test 1: Zeros consistency
    {
        cout << "Testing zeros consistency..." << endl;
        
        TensorOptions cpu_opts;
        cpu_opts.device = DeviceIndex(Device::CPU);
        cpu_opts.dtype = Dtype::Float32;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Float32;
        
        Tensor cpu_zeros = Tensor::zeros(Shape{{5, 5}}, cpu_opts);
        Tensor gpu_zeros = Tensor::zeros(Shape{{5, 5}}, gpu_opts);
        
        Tensor gpu_on_cpu = gpu_zeros.to_cpu();
        
        const float* cpu_data = static_cast<const float*>(cpu_zeros.data());
        const float* gpu_data = static_cast<const float*>(gpu_on_cpu.data());
        
        bool consistent = true;
        for (int64_t i = 0; i < cpu_zeros.numel(); ++i) {
            if (cpu_data[i] != gpu_data[i]) {
                cout << "Mismatch at index " << i << ": CPU=" << cpu_data[i] << ", GPU=" << gpu_data[i] << endl;
                consistent = false;
                break;
            }
        }
        
        assert(consistent);
        cout << "✓ Zeros consistency verified" << endl;
    }
    
    // Test 2: Ones consistency
    {
        cout << "Testing ones consistency..." << endl;
        
        TensorOptions cpu_opts;
        cpu_opts.device = DeviceIndex(Device::CPU);
        cpu_opts.dtype = Dtype::Int32;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Int32;
        
        Tensor cpu_ones = Tensor::ones(Shape{{10}}, cpu_opts);
        Tensor gpu_ones = Tensor::ones(Shape{{10}}, gpu_opts);
        
        Tensor gpu_on_cpu = gpu_ones.to_cpu();
        
        const int32_t* cpu_data = static_cast<const int32_t*>(cpu_ones.data());
        const int32_t* gpu_data = static_cast<const int32_t*>(gpu_on_cpu.data());
        
        bool consistent = true;
        for (int64_t i = 0; i < cpu_ones.numel(); ++i) {
            if (cpu_data[i] != gpu_data[i]) {
                consistent = false;
                break;
            }
        }
        
        assert(consistent);
        cout << "✓ Ones consistency verified" << endl;
    }
    
    // Test 3: Full consistency with different values
    {
        cout << "Testing full consistency with different values..." << endl;
        
        vector<float> test_values = {0.0f, 1.0f, -1.0f, 42.5f, -99.9f};
        
        for (float value : test_values) {
            TensorOptions cpu_opts;
            cpu_opts.device = DeviceIndex(Device::CPU);
            cpu_opts.dtype = Dtype::Float32;
            
            TensorOptions gpu_opts;
            gpu_opts.device = DeviceIndex(Device::CUDA);
            gpu_opts.dtype = Dtype::Float32;
            
            Tensor cpu_full = Tensor::full(Shape{{8}}, cpu_opts, value);
            Tensor gpu_full = Tensor::full(Shape{{8}}, gpu_opts, value);
            
            Tensor gpu_on_cpu = gpu_full.to_cpu();
            
            const float* cpu_data = static_cast<const float*>(cpu_full.data());
            const float* gpu_data = static_cast<const float*>(gpu_on_cpu.data());
            
            bool consistent = true;
            for (int64_t i = 0; i < cpu_full.numel(); ++i) {
                if (cpu_data[i] != gpu_data[i]) {
                    consistent = false;
                    break;
                }
            }
            
            assert(consistent);
            cout << "✓ Full consistency verified for value: " << value << endl;
        }
    }
    
    // Test 4: Random distributions should have similar statistics
    {
        cout << "Testing random distribution statistical consistency..." << endl;
        
        TensorOptions cpu_opts;
        cpu_opts.device = DeviceIndex(Device::CPU);
        cpu_opts.dtype = Dtype::Float32;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Float32;
        
        // Test uniform random
        Tensor cpu_rand = Tensor::rand(Shape{{5000}}, cpu_opts);
        Tensor gpu_rand = Tensor::rand(Shape{{5000}}, gpu_opts);
        
        Tensor gpu_rand_cpu = gpu_rand.to_cpu();
        
        const float* cpu_rand_data = static_cast<const float*>(cpu_rand.data());
        const float* gpu_rand_data = static_cast<const float*>(gpu_rand_cpu.data());
        
        // Calculate means for both distributions
        double cpu_sum = 0.0, gpu_sum = 0.0;
        for (int64_t i = 0; i < cpu_rand.numel(); ++i) {
            cpu_sum += cpu_rand_data[i];
            gpu_sum += gpu_rand_data[i];
        }
        
        double cpu_mean = cpu_sum / cpu_rand.numel();
        double gpu_mean = gpu_sum / gpu_rand.numel();
        
        cout << "Uniform random means - CPU: " << cpu_mean << ", GPU: " << gpu_mean << endl;
        
        // Means should both be around 0.5 for uniform [0,1)
        assert(std::abs(cpu_mean - 0.5) < 0.1);
        assert(std::abs(gpu_mean - 0.5) < 0.1);
        
        cout << "✓ Random distribution statistical consistency verified" << endl;
    }

    cout << "\nAll GPU-CPU factory consistency tests passed!" << endl;
}

int main ()
{
    test_gpu_factory_consistency();
}