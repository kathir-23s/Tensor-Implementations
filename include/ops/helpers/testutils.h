#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "dtype/DtypeCastUtils.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace TestUtils {
using namespace OwnTensor;
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
// In your test_utils.h or wherever verify_tensor_values is defined:

bool verify_tensor_values(const Tensor& tensor, const std::vector<float>& expected, double tolerance) {
    if (tensor.numel() != expected.size()) return false;
    
    // Move to CPU if needed
    Tensor cpu_tensor = tensor.is_cpu() ? tensor : tensor.to(DeviceIndex(Device::CPU));
    
    // Handle different dtypes
    if (cpu_tensor.dtype() == Dtype::Float32) {
        const float* data = cpu_tensor.data<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(data[i] - expected[i]) > tolerance) return false;
        }
        return true;
    } 
    else if (cpu_tensor.dtype() == Dtype::Float64) {
        const double* data = cpu_tensor.data<double>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(data[i] - expected[i]) > tolerance) return false;
        }
        return true;
    }
    // ADD INTEGER SUPPORT:
    else if (cpu_tensor.dtype() == Dtype::Int16) {
        const int16_t* data = cpu_tensor.data<int16_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(data[i] - static_cast<int16_t>(expected[i])) > tolerance) return false;
        }
        return true;
    }
    else if (cpu_tensor.dtype() == Dtype::Int32) {
        const int32_t* data = cpu_tensor.data<int32_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(data[i] - static_cast<int32_t>(expected[i])) > tolerance) return false;
        }
        return true;
    }
    else if (cpu_tensor.dtype() == Dtype::Int64) {
        const int64_t* data = cpu_tensor.data<int64_t>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(data[i] - static_cast<int64_t>(expected[i])) > tolerance) return false;
        }
        return true;
    }
    else if (cpu_tensor.dtype() == Dtype::Float16 || cpu_tensor.dtype() == Dtype::Bfloat16) {
        // Convert to float32 for comparison
        Tensor temp = convert_half_to_float32(cpu_tensor);
        const float* data = temp.data<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::fabs(data[i] - expected[i]) > tolerance) return false;
        }
        return true;
    }
    else {
        std::cout << "Unsupported dtype for verify_tensor_values\n";
        return false;
    }
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