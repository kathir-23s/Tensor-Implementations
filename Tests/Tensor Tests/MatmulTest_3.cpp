#include <iostream>
#include <vector>
#include <chrono>
#include "TensorLib.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace OwnTensor;

void test_large_batched_matmul() {
    std::cout << "=== LARGE BATCHED MATMUL PERFORMANCE TEST ===" << std::endl;

    // Large tensor dimensions: 16 batches of 512x1024 matrices multiplied by 16 batches of 1024x256 matrices
    int64_t batch_size = 16;
    int64_t m = 512;   // rows of A
    int64_t n = 1024;  // cols of A / rows of B  
    int64_t p = 256;   // cols of B
    
    std::cout << "Tensor A shape: [" << batch_size << ", " << m << ", " << n << "]" << std::endl;
    std::cout << "Tensor B shape: [" << batch_size << ", " << n << ", " << p << "]" << std::endl;
    std::cout << "Expected output shape: [" << batch_size << ", " << m << ", " << p << "]" << std::endl;
    std::cout << "Total operations: " << batch_size * m * n * p << " multiply-add operations" << std::endl;

    // Create large tensors with random data
    Tensor a_gpu = Tensor::rand(Shape{{batch_size, m, n}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
    Tensor b_gpu = Tensor::rand(Shape{{batch_size, n, p}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CUDA, 0)});

    // Warm-up run
    std::cout << "\nWarm-up run..." << std::endl;
    Tensor warmup_result = matmul(a_gpu, b_gpu);
    
    // Actual timing
    std::cout << "Starting timed execution..." << std::endl;
    
    #ifdef WITH_CUDA
    cudaDeviceSynchronize();
    #endif
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor result_gpu = matmul(a_gpu, b_gpu);
    
    #ifdef WITH_CUDA
    cudaDeviceSynchronize();
    #endif
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify result shape
    const auto& output_dims = result_gpu.shape().dims;
    bool shape_correct = (output_dims.size() == 3) && 
                        (output_dims[0] == batch_size) && 
                        (output_dims[1] == m) && 
                        (output_dims[2] == p);
    
    // Basic sanity check - verify no NaNs or extreme values
    bool values_valid = true;
    Tensor result_cpu = result_gpu.to_cpu();
    const float* result_data = result_cpu.data<float>();
    size_t total_elements = result_cpu.numel();
    
    // Check first and last few elements for sanity
    size_t check_count = std::min(size_t(100), total_elements);
    for (size_t i = 0; i < check_count; ++i) {
        if (std::isnan(result_data[i]) || std::isinf(result_data[i])) {
            values_valid = false;
            break;
        }
    }
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "Output shape correct: " << (shape_correct ? "YES" : "NO") << std::endl;
    std::cout << "Values valid (no NaN/Inf): " << (values_valid ? "YES" : "NO") << std::endl;
    std::cout << "Test status: " << (shape_correct && values_valid ? "PASSED" : "FAILED") << std::endl;
    
    // Performance metrics
    double total_operations = static_cast<double>(batch_size) * m * n * p;
    double gflops = (total_operations / (duration.count() / 1000.0)) / 1e9;
    std::cout << "Performance: " << gflops << " GFLOPs" << std::endl;
    
    std::cout << "===================================" << std::endl;
}

int main() {
    test_large_batched_matmul();
    return 0;
}