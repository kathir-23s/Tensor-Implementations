#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "core/Tensor.h"
#include "dtype/Types.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace TestUtils {

// ============================================================================
// Compare two numeric values with absolute tolerance
// ============================================================================
template<typename T1, typename T2>
bool compare_values(T1 a, T2 b, double tolerance = 1e-6) {
    return std::abs(static_cast<double>(a) - static_cast<double>(b)) < tolerance;
}

// ============================================================================
// Compare with relative tolerance (better for exp, log, etc.)
// ============================================================================
template<typename T1, typename T2>
bool compare_values_relative(T1 a, T2 b, double rel_tol = 1e-3, double abs_tol = 1e-6) {
    double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
    double max_val = std::max(std::abs(static_cast<double>(a)), 
                              std::abs(static_cast<double>(b)));
    
    // Use relative tolerance for large values, absolute for small values
    return diff <= std::max(rel_tol * max_val, abs_tol);
}

// ============================================================================
// Verify tensor values match expected values with adaptive tolerance
// ============================================================================
inline bool verify_tensor_values(
    const OwnTensor::Tensor& tensor,
    const std::vector<float>& expected,
    double tolerance = 1e-6
) {
    using namespace OwnTensor;
    
    // Adjust tolerance based on dtype - use relative tolerance for lower precision types
    double rel_tol = tolerance;
    double abs_tol = 1e-6;
    
    if (tensor.dtype() == Dtype::Bfloat16) {
        // Bfloat16 has low precision - use 5% relative tolerance minimum
        rel_tol = std::max(tolerance, 0.05);
        abs_tol = 1e-4;
    } else if (tensor.dtype() == Dtype::Float16) {
        // Float16 - use 1% relative tolerance minimum
        rel_tol = std::max(tolerance, 0.01);
        abs_tol = 1e-5;
    }
    
    // Check size matches
    if (tensor.numel() != expected.size()) {
        std::cerr << "Size mismatch: tensor has " << tensor.numel() 
                  << " elements, expected " << expected.size() << std::endl;
        return false;
    }
    
    // Copy tensor data to CPU if on GPU
    Tensor cpu_tensor = tensor;
    if (tensor.device().is_cuda()) {
        cpu_tensor = tensor.to(DeviceIndex(Device::CPU));
    }
    
    // Get data pointer based on dtype and verify with relative tolerance
    bool all_match = true;
    size_t first_mismatch = 0;
    bool found_mismatch = false;
    
    if (cpu_tensor.dtype() == Dtype::Float32) {
        const float* data = cpu_tensor.data<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!compare_values_relative(data[i], expected[i], rel_tol, abs_tol)) {
                if (!found_mismatch) {
                    first_mismatch = i;
                    found_mismatch = true;
                }
                all_match = false;
            }
        }
    } else if (cpu_tensor.dtype() == Dtype::Float64) {
        const double* data = cpu_tensor.data<double>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!compare_values_relative(data[i], expected[i], rel_tol, abs_tol)) {
                if (!found_mismatch) {
                    first_mismatch = i;
                    found_mismatch = true;
                }
                all_match = false;
            }
        }
    } else if (cpu_tensor.dtype() == Dtype::Float16) {
        const float16_t* data = cpu_tensor.data<float16_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!compare_values_relative(static_cast<float>(data[i]), expected[i], rel_tol, abs_tol)) {
                if (!found_mismatch) {
                    first_mismatch = i;
                    found_mismatch = true;
                }
                all_match = false;
            }
        }
    } else if (cpu_tensor.dtype() == Dtype::Bfloat16) {
        const bfloat16_t* data = cpu_tensor.data<bfloat16_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!compare_values_relative(static_cast<float>(data[i]), expected[i], rel_tol, abs_tol)) {
                if (!found_mismatch) {
                    first_mismatch = i;
                    found_mismatch = true;
                }
                all_match = false;
            }
        }
    } else {
        std::cerr << "Unsupported dtype for verify_tensor_values" << std::endl;
        return false;
    }
    
    // If there was a mismatch, print details about the first one
    if (!all_match) {
        std::cerr << "Verification failed. First mismatch at index " << first_mismatch 
                  << " (using rel_tol=" << rel_tol << ", abs_tol=" << abs_tol << ")" << std::endl;
    }
    
    return all_match;
}

// ============================================================================
// Create tensor from float vector with automatic dtype conversion
// ============================================================================
inline OwnTensor::Tensor create_tensor_from_float(
    const std::vector<float>& data,
    const OwnTensor::DeviceIndex& device,
    OwnTensor::Dtype dtype
) {
    using namespace OwnTensor;
    
    // Create tensor (allocates memory on correct device)
    Tensor tensor({{static_cast<int64_t>(data.size())}}, dtype, device);
    
    // For GPU tensors, we MUST copy via cudaMemcpy, not set_data()
    if (device.is_cuda()) {
#ifdef WITH_CUDA
        // Convert data to target dtype on CPU first
        if (dtype == Dtype::Int16) {
            std::vector<int16_t> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<int16_t>(), converted.data(), 
                      converted.size() * sizeof(int16_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Int32) {
            std::vector<int32_t> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<int32_t>(), converted.data(), 
                      converted.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Int64) {
            std::vector<int64_t> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<int64_t>(), converted.data(), 
                      converted.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Float32) {
            cudaMemcpy(tensor.data<float>(), data.data(), 
                      data.size() * sizeof(float), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Float64) {
            std::vector<double> converted(data.begin(), data.end());
            cudaMemcpy(tensor.data<double>(), converted.data(), 
                      converted.size() * sizeof(double), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Float16) {
            std::vector<float16_t> converted;
            for (float f : data) converted.push_back(float16_t(f));
            cudaMemcpy(tensor.data<float16_t>(), converted.data(), 
                      converted.size() * sizeof(float16_t), cudaMemcpyHostToDevice);
        } else if (dtype == Dtype::Bfloat16) {
            std::vector<bfloat16_t> converted;
            for (float f : data) converted.push_back(bfloat16_t(f));
            cudaMemcpy(tensor.data<bfloat16_t>(), converted.data(), 
                      converted.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice);
        }
        
        // CRITICAL: Synchronize after copy!
        cudaDeviceSynchronize();
#endif
    } else {
        // CPU path - use set_data() as before
        if (dtype == Dtype::Int16) {
            std::vector<int16_t> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Int32) {
            std::vector<int32_t> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Int64) {
            std::vector<int64_t> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Float32) {
            tensor.set_data(data);
        } else if (dtype == Dtype::Float64) {
            std::vector<double> converted(data.begin(), data.end());
            tensor.set_data(converted);
        } else if (dtype == Dtype::Float16) {
            std::vector<float16_t> converted;
            for (float f : data) converted.push_back(float16_t(f));
            tensor.set_data(converted);
        } else if (dtype == Dtype::Bfloat16) {
            std::vector<bfloat16_t> converted;
            for (float f : data) converted.push_back(bfloat16_t(f));
            tensor.set_data(converted);
        }
    }
    
    return tensor;
}

} // namespace TestUtils