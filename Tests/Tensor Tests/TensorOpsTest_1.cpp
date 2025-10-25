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

void test_addition(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR ADDITION TEST ===" << std::endl;
    


    // CPU addition
    std::cout << "\nCPU Addition..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor result_cpu = a_cpu + b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU addition
    std::cout << "\nGPU Addition..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    Tensor result_gpu = a_gpu + b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = result_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // result_cpu.display(std::cout, 7);
    std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(result_cpu, result_gpu_on_cpu)) {
        std::cout << "\nADDITION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nADDITION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

void test_subtraction(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR SUBTRACTION TEST ===" << std::endl;

    // CPU subtraction
    std::cout << "\nCPU Subtraction..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor result_cpu = a_cpu - b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU subtraction
    std::cout << "\nGPU Subtraction..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    Tensor result_gpu = a_gpu - b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = result_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // result_cpu.display(std::cout, 7);
    std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(result_cpu, result_gpu_on_cpu)) {
        std::cout << "\nSUBTRACTION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nSUBTRACTION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

void test_multiplication(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR MULTIPLICATION TEST ===" << std::endl;

    // CPU multiplication
    std::cout << "\nCPU Multiplication..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor result_cpu = a_cpu * b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU multiplication
    std::cout << "\nGPU Multiplication..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    Tensor result_gpu = a_gpu * b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = result_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // result_cpu.display(std::cout, 7);
    std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(result_cpu, result_gpu_on_cpu)) {
        std::cout << "\nMULTIPLICATION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nMULTIPLICATION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

void test_division(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== TENSOR DIVISION TEST ===" << std::endl;

    // CPU division
    std::cout << "\nCPU Division..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor result_cpu = a_cpu / b_cpu;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU division
    std::cout << "\nGPU Division..." << std::endl;
    Tensor a_gpu = a_cpu.to(DeviceIndex(Device::CUDA));
    Tensor b_gpu = b_cpu.to(DeviceIndex(Device::CUDA));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    Tensor result_gpu = a_gpu / b_gpu;
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = result_gpu.to_cpu();
    std::cout << "\n RESULTANT TENSORS ON CPU AND GPU: \n" << std::endl;
    // result_cpu.display(std::cout, 7);
    std::cout << "\n\n==========================================================================\n";
    // result_gpu_on_cpu.display(std::cout, 7);

    if (compare_tensors(result_cpu, result_gpu_on_cpu)) {
        std::cout << "\nDIVISION TEST PASSED" << std::endl;
    } else {
        std::cout << "\nDIVISION TEST FAILED" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;
}

int main() {
        // Create tensors
    Tensor a_cpu = Tensor::rand(Shape{{10000, 10000}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    Tensor b_cpu = Tensor::rand(Shape{{10000, 10000}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});




    test_addition(a_cpu, b_cpu);
    test_subtraction(a_cpu, b_cpu);
    test_multiplication(a_cpu, b_cpu);
    test_division(a_cpu, b_cpu);
    return 0;
}
