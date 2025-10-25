#include <iostream>
#include <vector>
#include "TensorLib.h"

using namespace OwnTensor;

void print_shape(const std::string& name, const Tensor& tensor) {
    std::cout << name << " shape: [";
    const auto& dims = tensor.shape().dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i];
        if (i < dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void test_batched_matmul() {
    std::cout << "=== BATCHED MATMUL TEST ===" << std::endl;

    // Simple test case: 2 batches of 2x3 matrices multiplied by 2 batches of 3x2 matrices
    std::vector<float> a_data = {
        // Batch 0
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        // Batch 1  
        2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f
    };

    std::vector<float> b_data = {
        1.0f, 4.0f, 
        5.0f, 5.0f,
        6.0f, 7.0f
    };

    // Create tensors: (2, 2, 3) and (2, 3, 2)
    Tensor a_cpu(Shape{{2, 1, 2, 3}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    Tensor b_cpu(Shape{{3, 2}}, TensorOptions{Dtype::Float32, DeviceIndex(Device::CPU)});
    
    a_cpu.set_data(a_data);
    b_cpu.set_data(b_data);

    print_shape("Input A: \n", a_cpu); a_cpu.display(std::cout, 5);
    print_shape("Input B: \n", b_cpu); b_cpu.display(std::cout, 5);

    // CPU batched matmul
    std::cout << "\nCPU Batched Matmul..." << std::endl;
    Tensor result_cpu = matmul(b_cpu, a_cpu);
    print_shape("Output", result_cpu);

    // Display results
    std::cout << "\nCPU Results:" << std::endl;
    result_cpu.display(std::cout, 4);

    // GPU batched matmul
    #ifdef WITH_CUDA
    std::cout << "\nGPU Batched Matmul..." << std::endl;
    Tensor a_gpu = a_cpu.to_cuda();
    Tensor b_gpu = b_cpu.to_cuda();
    
    Tensor result_gpu = matmul(b_gpu, a_gpu);
    Tensor result_gpu_cpu = result_gpu.to_cpu();
    
    std::cout << "GPU Results:" << std::endl;
    result_gpu_cpu.display(std::cout, 4);

    // Quick validation
    bool passed = true;
    if (result_cpu.shape().dims != result_gpu_cpu.shape().dims) {
        std::cout << "SHAPE MISMATCH!" << std::endl;
        passed = false;
    }
    
    if (passed) {
        std::cout << "\nBATCHED MATMUL TEST COMPLETED" << std::endl;
    }
    #else
    std::cout << "\nCUDA not available - GPU test skipped" << std::endl;
    #endif

    std::cout << "===================================" << std::endl;
}

int main() {
    test_batched_matmul();
    return 0;
}