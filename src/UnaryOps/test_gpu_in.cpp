#include <iostream>
#include <vector>
#include <cassert>

#include "../../include/Tensor.h"
#include "../../include/Types.h"
#include "../../include/TensorUnaryOps.hpp"  // high-level exp()

using namespace OwnTensor;

int main() {
    // Test data (CPU side, float32)
    std::vector<float> host_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

    // ==========================================
    // Test 1: Float32 baseline
    // ==========================================
    {
        std::cout << "\n--- Testing Float32 ---" << std::endl;
        Tensor input({{5}}, Dtype::Float32, Device::CUDA);
        input.set_data(host_data); // copies to GPU internally
        exp_(input); // should call exp_gpu_wrap()

        Tensor result = input.to_cpu();
        std::cout << "Output (Float32): ";
        result.display(std::cout, 5);
        std::cout << std::endl;
    }

    // ==========================================
    // Test 2: Float16
    // ==========================================
    {
        std::cout << "\n--- Testing Float16 ---" << std::endl;
        std::vector<float16_t> host_data_f16(host_data.size());
        for (size_t i = 0; i < host_data.size(); ++i)
            host_data_f16[i] = float16_t(host_data[i]);

        Tensor input({{5}}, Dtype::Float16, Device::CUDA);
        input.set_data(host_data_f16); // upload half data to GPU
        exp_(input); // should dispatch to unary_half_kernel_gpu

        Tensor result = input.to_cpu();  // get result back
        std::cout << "Output (Float16): ";
        result.display(std::cout, 5);
        std::cout << std::endl;
    }

    // ==========================================
    // Test 3: Bfloat16
    // ==========================================
    {
        std::cout << "\n--- Testing Bfloat16 ---" << std::endl;
        std::vector<bfloat16_t> host_data_bf16(host_data.size());
        for (size_t i = 0; i < host_data.size(); ++i)
            host_data_bf16[i] = bfloat16_t(host_data[i]);

        Tensor input({{5}}, Dtype::Bfloat16, Device::CUDA);
        input.set_data(host_data_bf16);
        exp_(input);

        Tensor result = input.to_cpu();
        std::cout << "Output (Bfloat16): ";
        result.display(std::cout, 5);
        std::cout << std::endl;
    }
    // ==========================================
    // Test 4: Int32
    // ==========================================
    {
        std::cout << "\n--- Testing Int32 ---" << std::endl;
        std::vector<int32_t> data {1,2,3,4,5};

        Tensor input({{5}}, Dtype::Int32, Device::CUDA);
        input.set_data(data);
        exp_(input);

        Tensor result = input.to_cpu();
        std::cout << "Output (Int32): ";
        result.display(std::cout, 5);
        std::cout << std::endl;
    }

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
