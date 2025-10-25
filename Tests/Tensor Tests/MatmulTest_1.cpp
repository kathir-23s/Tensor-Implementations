#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include "TensorLib.h"

using namespace OwnTensor;

bool compare_tensors(const Tensor& t1, const Tensor& t2, float tolerance = 1e-3f) {
    if (t1.numel() != t2.numel() || t1.dtype() != t2.dtype()) {
        return false;
    }
    
    return dispatch_by_dtype(t1.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* data1 = t1.data<T>();
        const T* data2 = t2.data<T>();
        size_t n = t1.numel();
        
        if constexpr (std::is_floating_point_v<T>) {
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(data1[i] - data2[i]) > static_cast<T>(tolerance)) {
                    std::cout << "Mismatch at index " << i 
                              << ": " << data1[i] << " vs " << data2[i] << std::endl;
                    return false;
                }
            }
        } else {
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
    Tensor cloned(tensor.shape(), tensor.dtype(), tensor.device());
    
    if (tensor.device().is_cpu()) {
        dispatch_by_dtype(tensor.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            const T* src_data = tensor.data<T>();
            T* dst_data = cloned.data<T>();
            size_t n = tensor.numel();
            std::copy(src_data, src_data + n, dst_data);
        });
    } else {
        #ifdef WITH_CUDA
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

void test_matmul(const Tensor& a_cpu, const Tensor& b_cpu) {
    std::cout << "=== MATRIX MULTIPLICATION TEST ===" << std::endl;

    #ifdef WITH_DISPLAY
    std::cout << "INPUTS: LHS\n" << std::endl;
    a_cpu.display(std::cout, 6);
    std::cout << "INPUTS: RHS\n" << std::endl;
    b_cpu.display(std::cout, 6);
    #endif

    // CPU matmul
    std::cout << "\nCPU Matrix Multiplication..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor result_cpu = matmul(a_cpu, b_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << "us" << std::endl;
    
    // GPU matmul
    std::cout << "\nGPU Matrix Multiplication..." << std::endl;
    Tensor a_gpu = a_cpu.to_cuda();
    Tensor b_gpu = b_cpu.to_cuda();
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    Tensor result_gpu = matmul(a_gpu, b_gpu);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << "us" << std::endl;
    
    // Compare results
    Tensor result_gpu_on_cpu = result_gpu.to_cpu();
    
    #ifdef WITH_DISPLAY
    std::cout << "RESULTS: \n" << std::endl;
    std::cout << "CPU:\n" << std::endl;
    result_cpu.display(std::cout, 6);
    std::cout << "CUDA:\n" << std::endl;
    result_gpu_on_cpu.display(std::cout, 6);
    #endif

    if (compare_tensors(result_cpu, result_gpu_on_cpu)) {
        std::cout << "\nMATMUL TEST PASSED" << std::endl;
    } else {
        std::cout << "\nMATMUL TEST FAILED" << std::endl;
    }
    
    std::cout << "===================================\n" << std::endl;
}

int main() {

    std::vector<float> source_data_a = {-1.0f, -2.0f, 3.0f, 4.0f, 5.0f, -6.0f,
                                    1.0f, 2.0f, -3.0f, -4.0f, -5.0f, 6.0f};

    std::vector<float> source_data_b = {0.0f,  4.0f, 5.0f, -6.0f,
                                    1.0f, 2.0f, -5.0f, 6.0f};

                                    
    // Test case 1: Small matrices
    // std::cout << "TEST CASE 1: Small matrices (3x4 * 4x2)" << std::endl;

    // Tensor a_small(Shape{{3, 4}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    // a_small.set_data(source_data_a);
    // Tensor b_small(Shape{{4, 2}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    // b_small.set_data(source_data_b);

    // test_matmul(a_small, b_small);

    // Test case 2: Medium matrices
    std::cout << "TEST CASE 2: Medium matrices (100x200 * 200x50)" << std::endl;
    Tensor a_medium = Tensor::rand(Shape{{100, 200}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    Tensor b_medium = Tensor::rand(Shape{{200, 50}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    test_matmul(a_medium, b_medium);

    // Test case 3: Medium matrices
    // {
    // std::cout << "TEST CASE 3: Medium matrices (1000x2000 * 2000x5000)" << std::endl;
    // Tensor a_medium = Tensor::rand(Shape{{1000, 2000}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    // Tensor b_medium = Tensor::rand(Shape{{2000, 5000}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    // test_matmul(a_medium, b_medium);
    // }


    // // Test case 4: Different data types
    // std::cout << "TEST CASE 4: Float64 matrices (5x3 * 3x7)" << std::endl;
    // Tensor a_double = Tensor::ones(Shape{{5, 3}}, TensorOptions{Dtype::Float64, DeviceIndex(Device::CPU)});
    // Tensor b_double = Tensor::ones(Shape{{6, 7}}, TensorOptions{Dtype::Float64, DeviceIndex(Device::CPU)});
    // test_matmul(a_double, b_double);

    return 0;
}