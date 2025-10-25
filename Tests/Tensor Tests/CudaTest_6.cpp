#include "TensorLib.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace std;
using namespace OwnTensor;

void test_gpu_rand_factory() 
{
    cout << "\n=== test_gpu_rand_factory ===\n" << endl;
    cout << "Testing random uniform distribution factory function directly on GPU..." << endl;
    
    // Test 1: Basic random uniform distribution on GPU
    {
        cout << "Testing basic random uniform initialization..." << endl;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Float32;
        
        Tensor gpu_rand = Tensor::rand(Shape{{1000}}, gpu_opts);
        
        cout << "GPU random tensor created, device: " << (gpu_rand.device().is_cuda() ? "CUDA" : "CPU") << endl;
        cout << "Transferring to CPU for verification..." << endl;
        
        Tensor cpu_verify = gpu_rand.to_cpu();
        const float* data = static_cast<const float*>(cpu_verify.data());
        
        // Verify values are in valid range [0, 1)
        bool all_valid = true;
        float min_val = 1.0f;
        float max_val = 0.0f;
        
        for (int64_t i = 0; i < gpu_rand.numel(); ++i) {
            if (data[i] < 0.0f || data[i] >= 1.0f) {
                cout << "Invalid value found at index " << i << ": " << data[i] << endl;
                all_valid = false;
                break;
            }
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        assert(all_valid);
        cout << "✓ GPU rand factory verified - all elements in range [0, 1)" << endl;
        cout << "  Range: [" << min_val << ", " << max_val << "]" << endl;
    }
    
    // Test 2: Random with different data types
    {
        cout << "Testing random with different data types..." << endl;
        
        vector<Dtype> test_dtypes = {Dtype::Float32, Dtype::Float64};
        
        for (Dtype dtype : test_dtypes) {
            TensorOptions opts;
            opts.device = DeviceIndex(Device::CUDA);
            opts.dtype = dtype;
            
            Tensor rand_tensor = Tensor::rand(Shape{{500}}, opts);
            Tensor cpu_tensor = rand_tensor.to_cpu();
            
            bool verified = true;
            if (dtype == Dtype::Float32) {
                const float* data = static_cast<const float*>(cpu_tensor.data());
                for (int64_t i = 0; i < rand_tensor.numel(); ++i) {
                    if (data[i] < 0.0f || data[i] >= 1.0f) {
                        verified = false;
                        break;
                    }
                }
            } else if (dtype == Dtype::Float64) {
                const double* data = static_cast<const double*>(cpu_tensor.data());
                for (int64_t i = 0; i < rand_tensor.numel(); ++i) {
                    if (data[i] < 0.0 || data[i] >= 1.0) {
                        verified = false;
                        break;
                    }
                }
            }
            
            assert(verified);
            cout << "✓ " << " dtype random uniform verified" << endl;
        }
    }
    
    // Test 3: Different shapes
    {
        cout << "Testing random with different shapes..." << endl;
        
        vector<Shape> test_shapes = {
            Shape{{10, 10}},
            Shape{{5, 5, 5}},
            Shape{{1000}},
            Shape{{2, 3, 4, 5}}
        };
        
        TensorOptions opts;
        opts.device = DeviceIndex(Device::CUDA);
        opts.dtype = Dtype::Float32;
        
        for (const auto& shape : test_shapes) {
            Tensor rand_tensor = Tensor::rand(shape, opts);
            Tensor cpu_tensor = rand_tensor.to_cpu();
            const float* data = static_cast<const float*>(cpu_tensor.data());
            
            bool valid = true;
            for (int64_t i = 0; i < rand_tensor.numel(); ++i) {
                if (data[i] < 0.0f || data[i] >= 1.0f) {
                    valid = false;
                    break;
                }
            }
            
            assert(valid);
            cout << "✓ Random shape " << " verified" << endl;
        }
    }

    cout << "\nAll GPU rand factory tests passed!" << endl;
}

void test_gpu_randn_factory() 
{
    cout << "\n=== test_gpu_randn_factory ===\n" << endl;
    cout << "Testing random normal distribution factory function directly on GPU..." << endl;
    
    // Test 1: Basic random normal distribution on GPU
    {
        cout << "Testing basic random normal initialization..." << endl;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Float32;
        
        Tensor gpu_randn = Tensor::randn(Shape{{1000}}, gpu_opts);
        
        cout << "GPU random normal tensor created, device: " << (gpu_randn.device().is_cuda() ? "CUDA" : "CPU") << endl;
        cout << "Transferring to CPU for verification..." << endl;
        
        Tensor cpu_verify = gpu_randn.to_cpu();
        const float* data = static_cast<const float*>(cpu_verify.data());
        
        // Calculate basic statistics to verify it's roughly normal
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (int64_t i = 0; i < gpu_randn.numel(); ++i) {
            sum += data[i];
            sum_sq += data[i] * data[i];
        }
        
        double mean = sum / gpu_randn.numel();
        double variance = (sum_sq / gpu_randn.numel()) - (mean * mean);
        double stddev = std::sqrt(variance);
        
        cout << "Normal distribution statistics:" << endl;
        cout << "  Mean: " << mean << " (expected: ~0.0)" << endl;
        cout << "  Stddev: " << stddev << " (expected: ~1.0)" << endl;
        
        // Verify reasonable values (not checking exact distribution, just sanity)
        assert(std::abs(mean) < 0.5);  // Mean should be close to 0
        assert(stddev > 0.5 && stddev < 1.5);  // Stddev should be close to 1
        
        cout << "✓ GPU randn factory verified - reasonable normal distribution" << endl;
    }
    
    // Test 2: Different data types for normal distribution
    {
        cout << "Testing random normal with different data types..." << endl;
        
        vector<Dtype> test_dtypes = {Dtype::Float32, Dtype::Float64};
        
        for (Dtype dtype : test_dtypes) {
            TensorOptions opts;
            opts.device = DeviceIndex(Device::CUDA);
            opts.dtype = dtype;
            
            Tensor randn_tensor = Tensor::randn(Shape{{800}}, opts);
            Tensor cpu_tensor = randn_tensor.to_cpu();
            
            // Basic validation that values are reasonable (not infinities/NaNs)
            bool valid = true;
            if (dtype == Dtype::Float32) {
                const float* data = static_cast<const float*>(cpu_tensor.data());
                for (int64_t i = 0; i < randn_tensor.numel(); ++i) {
                    if (!std::isfinite(data[i])) {
                        valid = false;
                        break;
                    }
                }
            } else if (dtype == Dtype::Float64) {
                const double* data = static_cast<const double*>(cpu_tensor.data());
                for (int64_t i = 0; i < randn_tensor.numel(); ++i) {
                    if (!std::isfinite(data[i])) {
                        valid = false;
                        break;
                    }
                }
            }
            
            assert(valid);
            cout << "✓ " << " dtype random normal verified" << endl;
        }
    }
    
    // Test 3: Large normal distribution
    {
        cout << "Testing large random normal tensor..." << endl;
        
        TensorOptions opts;
        opts.device = DeviceIndex(Device::CUDA);
        opts.dtype = Dtype::Float32;
        
        Tensor large_randn = Tensor::randn(Shape{{10000}}, opts);
        cout << "Large random normal tensor created: " << large_randn.numel() << " elements" << endl;
        
        Tensor cpu_sample = large_randn.to_cpu();
        const float* data = static_cast<const float*>(cpu_sample.data());
        
        // Check that values are finite and reasonable
        int finite_count = 0;
        for (int64_t i = 0; i < large_randn.numel(); ++i) {
            if (std::isfinite(data[i]) && std::abs(data[i]) < 10.0f) {
                finite_count++;
            }
        }
        
        // Should have mostly reasonable values
        assert(finite_count > large_randn.numel() * 0.99);
        cout << "✓ Large random normal tensor verified" << endl;
    }

    cout << "\nAll GPU randn factory tests passed!" << endl;
}

int main()
{
    test_gpu_rand_factory();
    test_gpu_randn_factory();
}