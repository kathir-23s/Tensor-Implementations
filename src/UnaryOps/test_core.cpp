#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "../../include/Tensor.h"
#include "../../include/Types.h"
#include "../../include/TensorUnaryOps.hpp"  // includes exp, exp_, log, log_, etc.

using namespace OwnTensor;

// -------------------------------------------
// Helper: Print tensor results after operation
// -------------------------------------------
void print_result(const std::string& test_name, const Tensor& t) {
    std::cout << test_name << ": ";
    t.display(std::cout, 1);
    std::cout << "\n";
}

// -------------------------------------------
// Test both in-place and out-of-place versions
// -------------------------------------------
void test_function_pair(
    const std::string& name,
    Tensor (*fn_out)(const Tensor&),
    void (*fn_inplace)(Tensor&)
) {
    std::cout << "\n=========== TESTING " << name << " ===========" << std::endl;

    std::vector<float> host_data = {0.0f, 1.0f, 2.0f, 3.5f, 5.0f};

    // ========== FLOAT32 CPU ==========
    {
        Tensor input({{5}}, Dtype::Float32, Device::CPU);
        input.set_data(host_data);
        Tensor out = fn_out(input);
        print_result(name + " (CPU Float32, Outplace)", out);

        fn_inplace(input);
        print_result(name + " (CPU Float32, Inplace)", input);
    }

    // ========== FLOAT32 GPU ==========
    {
        Tensor input({{5}}, Dtype::Float32, Device::CUDA);
        input.set_data(host_data);
        Tensor out = fn_out(input);
        Tensor out_cpu = out.to_cpu();
        print_result(name + " (GPU Float32, Outplace)", out_cpu);

        fn_inplace(input);
        Tensor in_cpu = input.to_cpu();
        print_result(name + " (GPU Float32, Inplace)", in_cpu);
    }

    // ========== FLOAT16 GPU ==========
    {
        std::vector<float16_t> host_data_f16(host_data.size());
        for (size_t i = 0; i < host_data.size(); ++i)
            host_data_f16[i] = float16_t(host_data[i]);

        Tensor input({{5}}, Dtype::Float16, Device::CUDA);
        input.set_data(host_data_f16);
        Tensor out = fn_out(input);
        Tensor out_cpu = out.to_cpu();
        print_result(name + " (GPU Float16, Outplace)", out_cpu);

        fn_inplace(input);
        Tensor in_cpu = input.to_cpu();
        print_result(name + " (GPU Float16, Inplace)", in_cpu);
    }

    // ========== BFLOAT16 GPU ==========
    {
        std::vector<bfloat16_t> host_data_bf16(host_data.size());
        for (size_t i = 0; i < host_data.size(); ++i)
            host_data_bf16[i] = bfloat16_t(host_data[i]);

        Tensor input({{5}}, Dtype::Bfloat16, Device::CUDA);
        input.set_data(host_data_bf16);
        Tensor out = fn_out(input);
        Tensor out_cpu = out.to_cpu();
        print_result(name + " (GPU Bfloat16, Outplace)", out_cpu);

        fn_inplace(input);
        Tensor in_cpu = input.to_cpu();
        print_result(name + " (GPU Bfloat16, Inplace)", in_cpu);
    }

    // ========== FLOAT16 CPU ==========
    {
        std::vector<float16_t> host_data_f16(host_data.size());
        for (size_t i = 0; i < host_data.size(); ++i)
            host_data_f16[i] = float16_t(host_data[i]);

        Tensor input({{5}}, Dtype::Float16, Device::CPU);
        input.set_data(host_data_f16);
        Tensor out = fn_out(input);
        Tensor out_cpu = out.to_cpu();
        print_result(name + " (GPU Float16, Outplace)", out_cpu);

        fn_inplace(input);
        Tensor in_cpu = input.to_cpu();
        print_result(name + " (GPU Float16, Inplace)", in_cpu);
    }

    // ========== BFLOAT16 CPU ==========
    {
        std::vector<bfloat16_t> host_data_bf16(host_data.size());
        for (size_t i = 0; i < host_data.size(); ++i)
            host_data_bf16[i] = bfloat16_t(host_data[i]);

        Tensor input({{5}}, Dtype::Bfloat16, Device::CPU);
        input.set_data(host_data_bf16);
        Tensor out = fn_out(input);
        Tensor out_cpu = out.to_cpu();
        print_result(name + " (CPU Bfloat16, Outplace)", out_cpu);

        fn_inplace(input);
        Tensor in_cpu = input.to_cpu();
        print_result(name + " (CPU Bfloat16, Inplace)", in_cpu);
    }

    // ========== INT32 CPU (should throw or warn) ==========
    {
        std::vector<int32_t> host_data_i32 = {0, 1, 2, 3, 4};
        Tensor input({{5}}, Dtype::Int32, Device::CPU);
        input.set_data(host_data_i32);

        try {
            Tensor out = fn_out(input);
            print_result(name + " (Int32, Outplace)", out);
        } catch (const std::exception& e) {
            std::cout << "Caught expected exception (Outplace): " << e.what() << std::endl;
        }
    }
    // ========== INT32 GPU (should throw or warn) ==========
    {
        std::vector<int32_t> host_data_i32 = {0, 1, 2, 3, 4};
        Tensor input({{5}}, Dtype::Int32, Device::CUDA);
        input.set_data(host_data_i32);

        try {
            Tensor out = fn_out(input);
            Tensor out_cpu = out.to_cpu();
            print_result(name + " (Int32, Outplace)", out_cpu);
        } catch (const std::exception& e) {
            std::cout << "Caught expected exception (Outplace): " << e.what() << std::endl;
        }

        try {
            fn_inplace(input);
            Tensor in_cpu = input.to_cpu();
            print_result(name + " (Int32, Inplace)", in_cpu);
        } catch (const std::exception& e) {
            std::cout << "Caught expected exception (Inplace): " << e.what() << std::endl;
        }
    }

    std::cout << "=========== END " << name << " ===========\n";
}

// -------------------------------------------
// MAIN TEST DRIVER
// -------------------------------------------
int main() {
    std::cout << "\n======== BEGIN UNARY OP TESTS ========\n";

    // Each pair: (outplace_fn, inplace_fn)
    test_function_pair("exp",   exp,   exp_);
    test_function_pair("exp2",  exp2,  exp2_);
    test_function_pair("log",   log,   log_);
    test_function_pair("log2",  log2,  log2_);
    test_function_pair("log10", log10, log10_);

    std::cout << "\n======== ALL UNARY OP TESTS COMPLETED ========\n";
    return 0;
}