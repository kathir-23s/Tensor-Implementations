#include <iostream>
#include <iomanip>
#include <chrono>
#include "TensorLib.h"

using namespace OwnTensor;

bool compare_tensors(const Tensor& t1, const Tensor& t2, float tolerance = 1e-7f) {
    if (t1.numel() != t2.numel() || t1.dtype() != t2.dtype()) {
        return false;
    }
    
    return dispatch_by_dtype(t1.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* data1 = t1.data<T>();
        const T* data2 = t2.data<T>();
        size_t n = t1.numel();
        
        if constexpr (std::is_floating_point_v<T>) {
            // Floating point comparison with tolerance
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(data1[i] - data2[i]) > static_cast<T>(tolerance)) {
                    std::cout << "Mismatch at index " << i 
                              << ": " << data1[i] << " vs " << data2[i] << std::endl;
                    return false;
                }
            }
        } else {
            // Exact comparison for integers
            for (size_t i = 0; i < n; ++i) {
                if (data1[i] != data2[i]) {
                    std::cout << "Mismatch at index " << i 
                              << ": " << data1[i] << " vs " << data2[i] << std::endl;
                    return false;
                }
            }
        }
        return true;
    });
}

Tensor clone(const Tensor& tensor) {
    // Create a new tensor with same properties
    Tensor cloned(tensor.shape(), tensor.dtype(), tensor.device());
    
    // Copy the data
    if (tensor.device().is_cpu()) {
        // CPU copy
        dispatch_by_dtype(tensor.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            const T* src_data = tensor.data<T>();
            T* dst_data = cloned.data<T>();
            size_t n = tensor.numel();
            std::copy(src_data, src_data + n, dst_data);
        });
    } else {
        // GPU copy - use your existing CUDA copy mechanisms
        #ifdef WITH_CUDA
            // If you have a cuda_copy_tensor function, use it
            // cuda_copy_tensor(tensor, cloned);
            // Otherwise, fall back to CPU copy via to_cpu()
            Tensor temp_cpu = tensor.to_cpu();
            dispatch_by_dtype(tensor.dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                const T* src_data = temp_cpu.data<T>();
                T* dst_data = cloned.data<T>();
                size_t n = temp_cpu.numel();
                std::copy(src_data, src_data + n, dst_data);
            });
        #else
            throw std::runtime_error("CUDA support not compiled");
        #endif
    }
    
    return cloned;
}

void test_add_inplace(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR IN-PLACE ADDITION TEST ===" << std::endl;

    // CPU in-place addition
    std::cout << "\nCPU In-place Addition..." << std::endl;
    Tensor a_cpu_copy1 = clone(a_cpu);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    a_cpu_copy1 += b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU in-place addition
    std::cout << "\nGPU In-place Addition..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    a_gpu += b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = a_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // a_cpu_copy1.display(std::cout, 7);
    // std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(a_cpu_copy1, result_gpu_on_cpu)) {
        std::cout << "\nIN-PLACE ADDITION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nIN-PLACE ADDITION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

void test_sub_inplace(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR IN-PLACE SUBTRACTION TEST ===" << std::endl;

    // CPU in-place subtraction
    std::cout << "\nCPU In-place Subtraction..." << std::endl;
    Tensor a_cpu_copy1 = clone(a_cpu);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    a_cpu_copy1 -= b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU in-place subtraction
    std::cout << "\nGPU In-place Subtraction..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    a_gpu -= b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = a_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // a_cpu_copy1.display(std::cout, 7);
    // std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(a_cpu_copy1, result_gpu_on_cpu)) {
        std::cout << "\nIN-PLACE SUBTRACTION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nIN-PLACE SUBTRACTION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

void test_mul_inplace(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR IN-PLACE MULTIPLICATION TEST ===" << std::endl;

    // CPU in-place multiplication
    std::cout << "\nCPU In-place Multiplication..." << std::endl;
    Tensor a_cpu_copy1 = clone(a_cpu);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    a_cpu_copy1 *= b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU in-place multiplication
    std::cout << "\nGPU In-place Multiplication..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    a_gpu *= b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = a_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // a_cpu_copy1.display(std::cout, 7);
    // std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(a_cpu_copy1, result_gpu_on_cpu)) {
        std::cout << "\nIN-PLACE MULTIPLICATION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nIN-PLACE MULTIPLICATION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

void test_div_inplace(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR IN-PLACE DIVISION TEST ===" << std::endl;

    // CPU in-place division
    std::cout << "\nCPU In-place Division..." << std::endl;
    Tensor a_cpu_copy1 = clone(a_cpu);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    a_cpu_copy1 /= b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU in-place division
    std::cout << "\nGPU In-place Division..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    a_gpu /= b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = a_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // a_cpu_copy1.display(std::cout, 7);
    // std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(a_cpu_copy1, result_gpu_on_cpu)) {
        std::cout << "\nIN-PLACE DIVISION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nIN-PLACE DIVISION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}


int main() {
        // Create tensors
    Tensor a_cpu = Tensor::rand(Shape{{10000, 10000}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    Tensor b_cpu = Tensor::rand(Shape{{10000, 10000}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});



    std::cout << "\n INPUT TENSORS A AND B: \n" << std::endl;
    // a_cpu.display(std::cout, 7);
    // std::cout << "\n---------------------------------------------\n";
    // b_cpu.display(std::cout, 7);


    test_add_inplace(a_cpu, b_cpu);
    test_sub_inplace(a_cpu, b_cpu);
    test_mul_inplace(a_cpu, b_cpu);
    test_div_inplace(a_cpu, b_cpu);
    return 0;
}
