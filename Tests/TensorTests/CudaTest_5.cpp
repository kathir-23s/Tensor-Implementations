#include "TensorLib.h"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace std;
using namespace OwnTensor;

void test_gpu_zeros_factory() 
{
    cout << "\n=== test_gpu_zeros_factory ===\n" << endl;
    cout << "Testing zeros factory function directly on GPU..." << endl;
    
    // Test 1: Basic zeros on GPU
    {
        cout << "Testing basic zeros initialization..." << endl;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Float32;
        
        Tensor gpu_zeros = Tensor::zeros(Shape{{3, 4}}, gpu_opts);
        
        cout << "GPU zeros tensor created, device: " << (gpu_zeros.device().is_cuda() ? "CUDA" : "CPU") << endl;
        cout << "Transferring to CPU for verification..." << endl;
        
        Tensor cpu_verify = gpu_zeros.to_cpu();
        const float* data = static_cast<const float*>(cpu_verify.data());
        
        bool all_zeros = true;
        for (int64_t i = 0; i < gpu_zeros.numel(); ++i) {
            if (data[i] != 0.0f) {
                cout << "Non-zero value found at index " << i << ": " << data[i] << endl;
                all_zeros = false;
                break;
            }
        }
        
        assert(all_zeros);
        cout << "✓ GPU zeros factory verified - all elements are zero" << endl;
    }
    
    // Test 2: Zeros with multiple data types
    {
        cout << "Testing zeros with multiple data types..." << endl;
        
        vector<Dtype> test_dtypes = {Dtype::Int16, Dtype::Int32, Dtype::Int64, Dtype::Float32, Dtype::Float64};
        
        for (Dtype dtype : test_dtypes) {
            TensorOptions opts;
            opts.device = DeviceIndex(Device::CUDA);
            opts.dtype = dtype;
            
            Tensor zeros_tensor = Tensor::zeros(Shape{{5}}, opts);
            Tensor cpu_tensor = zeros_tensor.to_cpu();
            
            bool verified = true;
            switch (dtype) {
                case Dtype::Int16: {
                    const int16_t* data = static_cast<const int16_t*>(cpu_tensor.data());
                    for (int64_t i = 0; i < zeros_tensor.numel(); ++i) {
                        if (data[i] != 0) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Int32: {
                    const int32_t* data = static_cast<const int32_t*>(cpu_tensor.data());
                    for (int64_t i = 0; i < zeros_tensor.numel(); ++i) {
                        if (data[i] != 0) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Int64: {
                    const int64_t* data = static_cast<const int64_t*>(cpu_tensor.data());
                    for (int64_t i = 0; i < zeros_tensor.numel(); ++i) {
                        if (data[i] != 0) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Float32: {
                    const float* data = static_cast<const float*>(cpu_tensor.data());
                    for (int64_t i = 0; i < zeros_tensor.numel(); ++i) {
                        if (data[i] != 0.0f) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Float64: {
                    const double* data = static_cast<const double*>(cpu_tensor.data());
                    for (int64_t i = 0; i < zeros_tensor.numel(); ++i) {
                        if (data[i] != 0.0) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                default:
                    verified = false;
                    cout << "Unsupported dtype for zeros test: " << static_cast<int>(dtype) << endl;
            }
            
            assert(verified);
            cout << "✓ " << static_cast<int>(dtype) << " dtype zeros verified" << endl;
        }
    }
    
    // Test 3: Large zeros tensor
    {
        cout << "Testing large zeros tensor..." << endl;
        
        TensorOptions opts;
        opts.device = DeviceIndex(Device::CUDA);
        opts.dtype = Dtype::Float32;
        
        Tensor large_zeros = Tensor::zeros(Shape{{10000, 100}}, opts);
        cout << "Large zeros tensor created: " << large_zeros.numel() << " elements" << endl;
        
        // Verify sample of elements
        Tensor cpu_sample = large_zeros.to_cpu();
        const float* data = static_cast<const float*>(cpu_sample.data());
        
        // Check first, middle, and last elements
        assert(data[0] == 0.0f);
        assert(data[500000] == 0.0f); // middle
        assert(data[999999] == 0.0f); // last
        
        cout << "✓ Large zeros tensor verified" << endl;
    }

    cout << "\nAll GPU zeros factory tests passed!" << endl;
}

void test_gpu_ones_factory() 
{
    cout << "\n=== test_gpu_ones_factory ===\n" << endl;
    cout << "Testing ones factory function directly on GPU..." << endl;
    
    // Test 1: Basic ones on GPU
    {
        cout << "Testing basic ones initialization..." << endl;
        
        TensorOptions gpu_opts;
        gpu_opts.device = DeviceIndex(Device::CUDA);
        gpu_opts.dtype = Dtype::Float32;
        
        Tensor gpu_ones = Tensor::ones(Shape{{2, 3}}, gpu_opts);
        
        cout << "GPU ones tensor created, device: " << (gpu_ones.device().is_cuda() ? "CUDA" : "CPU") << endl;
        
        Tensor cpu_verify = gpu_ones.to_cpu();
        const float* data = static_cast<const float*>(cpu_verify.data());
        
        bool all_ones = true;
        for (int64_t i = 0; i < gpu_ones.numel(); ++i) {
            if (data[i] != 1.0f) {
                cout << "Non-one value found at index " << i << ": " << data[i] << endl;
                all_ones = false;
                break;
            }
        }
        
        assert(all_ones);
        cout << "✓ GPU ones factory verified - all elements are one" << endl;
    }
    
    // Test 2: Ones with multiple data types
    {
        cout << "Testing ones with multiple data types..." << endl;
        
        vector<Dtype> test_dtypes = {Dtype::Int16, Dtype::Int32, Dtype::Int64, Dtype::Float32, Dtype::Float64};
        
        for (Dtype dtype : test_dtypes) {
            TensorOptions opts;
            opts.device = DeviceIndex(Device::CUDA);
            opts.dtype = dtype;
            
            Tensor ones_tensor = Tensor::ones(Shape{{4}}, opts);
            Tensor cpu_tensor = ones_tensor.to_cpu();
            
            bool verified = true;
            switch (dtype) {
                case Dtype::Int16: {
                    const int16_t* data = static_cast<const int16_t*>(cpu_tensor.data());
                    for (int64_t i = 0; i < ones_tensor.numel(); ++i) {
                        if (data[i] != 1) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Int32: {
                    const int32_t* data = static_cast<const int32_t*>(cpu_tensor.data());
                    for (int64_t i = 0; i < ones_tensor.numel(); ++i) {
                        if (data[i] != 1) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Int64: {
                    const int64_t* data = static_cast<const int64_t*>(cpu_tensor.data());
                    for (int64_t i = 0; i < ones_tensor.numel(); ++i) {
                        if (data[i] != 1) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Float32: {
                    const float* data = static_cast<const float*>(cpu_tensor.data());
                    for (int64_t i = 0; i < ones_tensor.numel(); ++i) {
                        if (data[i] != 1.0f) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Float64: {
                    const double* data = static_cast<const double*>(cpu_tensor.data());
                    for (int64_t i = 0; i < ones_tensor.numel(); ++i) {
                        if (data[i] != 1.0) {
                            verified = false;
                            break;
                        }
                    }
                    break;
                }
                default:
                    verified = false;
                    cout << "Unsupported dtype for ones test: " << static_cast<int>(dtype) << endl;
            }
            
            assert(verified);
            cout << "✓ " << static_cast<int>(dtype) << " dtype ones verified" << endl;
        }
    }

    cout << "\nAll GPU ones factory tests passed!" << endl;
}



void test_gpu_full_factory() 
{
    cout << "\n=== test_gpu_full_factory ===\n" << endl;
    cout << "Testing full factory function directly on GPU..." << endl;
    
    // Test 1: Full with multiple data types and values
    {
        cout << "Testing full with multiple data types and values..." << endl;
        
        vector<pair<Dtype, double>> test_cases = {
            {Dtype::Int16, 15},
            {Dtype::Int32, 42},
            {Dtype::Int64, -100},
            {Dtype::Float32, 7.5f},
            {Dtype::Float64, -3.14}
        };
        
        for (const auto& test_case : test_cases) {
            Dtype dtype = test_case.first;
            double fill_value = test_case.second;
            
            TensorOptions opts;
            opts.device = DeviceIndex(Device::CUDA);
            opts.dtype = dtype;
            
            Tensor gpu_full = Tensor::full(Shape{{3, 2}}, opts, fill_value);
            
            cout << "GPU full tensor created with dtype " << static_cast<int>(dtype) << " and value: " << fill_value << endl;
            
            Tensor cpu_verify = gpu_full.to_cpu();
            bool all_match = true;
            
            switch (dtype) {
                case Dtype::Int16: {
                    const int16_t* data = static_cast<const int16_t*>(cpu_verify.data());
                    int16_t expected = static_cast<int16_t>(fill_value);
                    for (int64_t i = 0; i < gpu_full.numel(); ++i) {
                        if (data[i] != expected) {
                            cout << "Mismatch at index " << i << ": expected " << expected << ", got " << data[i] << endl;
                            all_match = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Int32: {
                    const int32_t* data = static_cast<const int32_t*>(cpu_verify.data());
                    int32_t expected = static_cast<int32_t>(fill_value);
                    for (int64_t i = 0; i < gpu_full.numel(); ++i) {
                        if (data[i] != expected) {
                            cout << "Mismatch at index " << i << ": expected " << expected << ", got " << data[i] << endl;
                            all_match = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Int64: {
                    const int64_t* data = static_cast<const int64_t*>(cpu_verify.data());
                    int64_t expected = static_cast<int64_t>(fill_value);
                    for (int64_t i = 0; i < gpu_full.numel(); ++i) {
                        if (data[i] != expected) {
                            cout << "Mismatch at index " << i << ": expected " << expected << ", got " << data[i] << endl;
                            all_match = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Float32: {
                    const float* data = static_cast<const float*>(cpu_verify.data());
                    float expected = static_cast<float>(fill_value);
                    for (int64_t i = 0; i < gpu_full.numel(); ++i) {
                        if (std::abs(data[i] - expected) > 1e-5f) {
                            cout << "Mismatch at index " << i << ": expected " << expected << ", got " << data[i] << endl;
                            all_match = false;
                            break;
                        }
                    }
                    break;
                }
                case Dtype::Float64: {
                    const double* data = static_cast<const double*>(cpu_verify.data());
                    for (int64_t i = 0; i < gpu_full.numel(); ++i) {
                        if (std::abs(data[i] - fill_value) > 1e-6) {
                            cout << "Mismatch at index " << i << ": expected " << fill_value << ", got " << data[i] 
                                 << " (diff: " << std::abs(data[i] - fill_value) << ")" << endl;
                            all_match = false;
                            break;
                        }
                    }
                    break;
                }
                default:
                    all_match = false;
                    cout << "Unsupported dtype for full test: " << static_cast<int>(dtype) << endl;
            }
            
            if (!all_match) {
                cout << "FAILED: Full factory test for dtype " << static_cast<int>(dtype) << " with value " << fill_value << endl;
                // Print first few elements for debugging
                cout << "First 3 elements: ";
                switch (dtype) {
                    case Dtype::Float64: {
                        const double* data = static_cast<const double*>(cpu_verify.data());
                        for (int i = 0; i < 3 && i < gpu_full.numel(); ++i) {
                            cout << data[i] << " ";
                        }
                        break;
                    }
                    case Dtype::Float32: {
                        const float* data = static_cast<const float*>(cpu_verify.data());
                        for (int i = 0; i < 3 && i < gpu_full.numel(); ++i) {
                            cout << data[i] << " ";
                        }
                        break;
                    }
                }
                cout << endl;
            }
            
            assert(all_match);
            cout << "✓ Full factory verified for dtype " << static_cast<int>(dtype) << " with value " << fill_value << endl;
        }
    }

    cout << "\nAll GPU full factory tests passed!" << endl;
}

int main()
{
    test_gpu_zeros_factory();
    test_gpu_ones_factory();
    test_gpu_full_factory();
    
    cout << "\n=== All GPU factory tests completed successfully! ===" << endl;
    return 0;
}