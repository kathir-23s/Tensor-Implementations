#include "Tensor.h"
#include "Reduction.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#ifdef WITH_CUDA
#include <cuda_runtime.h>  // Add this line
#endif
using namespace OwnTensor;

void print_test_header(const std::string& test_name) {
    std::cout << "\n========================================\n";
    std::cout << "  " << test_name << "\n";
    std::cout << "========================================\n";
}

bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

void test_sum_reduction_gpu() {
    print_test_header("GPU Sum Reduction Test");
    
    Tensor cpu_tensor({{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor (CPU):\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    std::cout << "\nTensor transferred to GPU\n";
    
    // Test axis 0
    Tensor result_gpu = reduce_sum(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nSum along axis 0:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [15, 18, 21, 24]\n";
    
    // Test axis 1
    result_gpu = reduce_sum(gpu_tensor, {1}, false);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nSum along axis 1:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [10, 26, 42]\n";
    
    // Full reduction
    result_gpu = reduce_sum(gpu_tensor, {}, false);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nFull sum:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [78]\n";
}

void test_mean_reduction_gpu() {
    print_test_header("GPU Mean Reduction Test");
    
    Tensor cpu_tensor({{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor:\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_mean(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nMean along axis 0:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [2.5, 3.5, 4.5]\n";
    
    result_gpu = reduce_mean(gpu_tensor, {1}, false);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nMean along axis 1:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [2.0, 5.0]\n";
}

void test_minmax_reduction_gpu() {
    print_test_header("GPU Min/Max Reduction Test");
    
    Tensor cpu_tensor({{3, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {9, 2, 7, 4, 5, 6, 1, 8, 3};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor:\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_min(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nMin along axis 0:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [1, 2, 3]\n";
    
    result_gpu = reduce_max(gpu_tensor, {1}, false);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nMax along axis 1:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [9, 6, 8]\n";
}

void test_argminmax_reduction_gpu() {
    print_test_header("GPU ArgMin/ArgMax Reduction Test");
    
    Tensor cpu_tensor({{2, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 5.0f, 6.0f, 7.0f};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor:\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_argmin(gpu_tensor, {1}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nArgMin along axis 1:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [1, 1]\n";
    
    result_gpu = reduce_argmax(gpu_tensor, {1}, false);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nArgMax along axis 1:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [2, 0]\n";
}

void test_product_reduction_gpu() {
    print_test_header("GPU Product Reduction Test");
    
    Tensor cpu_tensor({{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor:\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_product(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nProduct along axis 0:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [4, 10, 18]\n";
}

void test_nan_aware_reduction_gpu() {
    print_test_header("GPU NaN-Aware Reduction Test");
    
    Tensor cpu_tensor({{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {1.0f, NAN, 3.0f, 4.0f, 5.0f, NAN};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor (with NaNs):\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_nansum(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nNaN-aware sum along axis 0:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [5, 5, 3] (ignoring NaNs)\n";
    
    result_gpu = reduce_nanmean(gpu_tensor, {1}, false);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nNaN-aware mean along axis 1:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [2, 4.5]\n";
}

void test_keepdim_gpu() {
    print_test_header("GPU Keepdim Test");
    
    Tensor cpu_tensor({{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i + 1);
    cpu_tensor.set_data(data);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    // Without keepdim
    Tensor result_gpu = reduce_sum(gpu_tensor, {1}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "Sum along axis 1 (keepdim=false):\n";
    std::cout << "Shape: [";
    for (auto d : result_cpu.shape().dims) std::cout << d << " ";
    std::cout << "]\n";
    std::cout << "Expected shape: [2, 4]\n";
    
    // With keepdim
    result_gpu = reduce_sum(gpu_tensor, {1}, true);
    result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nSum along axis 1 (keepdim=true):\n";
    std::cout << "Shape: [";
    for (auto d : result_cpu.shape().dims) std::cout << d << " ";
    std::cout << "]\n";
    std::cout << "Expected shape: [2, 1, 4]\n";
}

void test_multi_axis_reduction_gpu() {
    print_test_header("GPU Multi-Axis Reduction Test");
    
    Tensor cpu_tensor({{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i + 1);
    cpu_tensor.set_data(data);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_sum(gpu_tensor, {0, 2}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "Sum along axes [0, 2]:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "\nShape: [";
    for (auto d : result_cpu.shape().dims) std::cout << d << " ";
    std::cout << "]\n";
    std::cout << "Expected shape: [3]\n";
}

void test_integer_reduction_gpu() {
    print_test_header("GPU Integer Reduction Test");
    
    Tensor cpu_tensor({{2, 3}}, TensorOptions().with_dtype(Dtype::Int32));
    std::vector<int32_t> data = {1, 2, 3, 4, 5, 6};
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor (Int32):\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_sum(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nSum along axis 0:\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [5, 7, 9]\n";
    std::cout << "Output dtype: Int64\n";
}

void test_fp16_reduction_gpu() {
    print_test_header("GPU FP16 Reduction Test");
    
    Tensor cpu_tensor({{2, 3}}, TensorOptions().with_dtype(Dtype::Float16));
    std::vector<float16_t> data = {
        float16_t(1.0f), float16_t(2.0f), float16_t(3.0f),
        float16_t(4.0f), float16_t(5.0f), float16_t(6.0f)
    };
    cpu_tensor.set_data(data);
    
    std::cout << "Input tensor (Float16):\n";
    cpu_tensor.display(std::cout, 2);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    Tensor result_gpu = reduce_mean(gpu_tensor, {0}, false);
    Tensor result_cpu = result_gpu.to_cpu();
    
    std::cout << "\nMean along axis 0 (Float16):\n";
    result_cpu.display(std::cout, 2);
    std::cout << "Expected: [2.5, 3.5, 4.5]\n";
}

void test_large_tensor_reduction_gpu() {
    print_test_header("GPU Large Tensor Reduction Test");
    
    const int size = 1024;
    Tensor cpu_tensor({{size, size}}, TensorOptions().with_dtype(Dtype::Float32));
    
    std::vector<float> data(size * size, 1.0f);
    cpu_tensor.set_data(data);
    
    std::cout << "Input: " << size << "x" << size << " tensor of ones\n";
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor result_gpu = reduce_sum(gpu_tensor, {0}, false);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    Tensor result_cpu = result_gpu.to_cpu();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\nGPU reduction time: " << duration.count() << " ms\n";
    std::cout << "First 5 elements: ";
    const float* result_data = result_cpu.data<float>();
    for (int i = 0; i < 5; ++i) {
        std::cout << result_data[i] << " ";
    }
    std::cout << "\nExpected: [1024, 1024, 1024, ...]\n";
}
void test_fp16_numerical_accuracy() {
    print_test_header("FP16 Numerical Accuracy Test");
    
    // Test that FP16 doesn't overflow during accumulation
    Tensor cpu_tensor({{100}}, TensorOptions().with_dtype(Dtype::Float16));
    std::vector<float16_t> data(100, float16_t(600.0f));  // 100 × 600 = 60,000 (< 65504)
    cpu_tensor.set_data(data);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    Tensor result = reduce_sum(gpu_tensor, {}, false).to_cpu();
    
    float16_t result_val = result.data<float16_t>()[0];
    float result_float = static_cast<float>(result_val);
    
    std::cout << "Sum of 100 × 600.0 (FP16): " << result_float << "\n";
    std::cout << "Expected: 60000.0\n";
    std::cout << "Error: " << std::abs(result_float - 60000.0f) << "\n";
    
    // Check for overflow to infinity
    if (std::isinf(result_float)) {
        std::cerr << "❌ FAIL: Result overflowed to infinity!\n";
    } else if (std::abs(result_float - 60000.0f) > 100.0f) {
        std::cerr << "⚠️  WARNING: Large error detected\n";
    } else {
        std::cout << "✅ PASS: Accumulation is correct\n";
    }
}

void test_nan_aware_reductions_gpu() {
    print_test_header("GPU NaN-Aware Reductions Test");
    
    Tensor cpu_tensor({{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    std::vector<float> data = {1.0f, NAN, 3.0f, 4.0f, 5.0f, NAN};
    cpu_tensor.set_data(data);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    
    // Test NaN-aware sum
    Tensor result_sum = reduce_nansum(gpu_tensor, {}, false).to_cpu();
    float sum_val = result_sum.data<float>()[0];
    std::cout << "NaN-aware sum: " << sum_val << " (expected: 13.0)\n";
    
    // Test NaN-aware mean
    Tensor result_mean = reduce_nanmean(gpu_tensor, {0}, false).to_cpu();
    std::cout << "NaN-aware mean along axis 0:\n";
    result_mean.display(std::cout, 2);
    std::cout << "Expected: [2.5, 5.0, 3.0]\n";
    
    // Test NaN-aware argmax
    Tensor result_argmax = reduce_nanargmax(gpu_tensor, {1}, false).to_cpu();
    std::cout << "NaN-aware argmax along axis 1:\n";
    result_argmax.display(std::cout, 2);
    std::cout << "Expected: [2, 1] (indices of max, ignoring NaNs)\n";
}

void test_bf16_vs_fp16_comparison() {
    print_test_header("BF16 vs FP16 Precision Comparison");
    
    // BF16 advantage: larger range
    std::cout << "Testing large numbers (BF16 advantage):\n";
    Tensor bf16_tensor({{3}}, TensorOptions().with_dtype(Dtype::Bfloat16));
    std::vector<bfloat16_t> bf16_data = {
        bfloat16_t(1e10f), 
        bfloat16_t(2e10f), 
        bfloat16_t(3e10f)
    };
    bf16_tensor.set_data(bf16_data);
    
    Tensor gpu_bf16 = bf16_tensor.to_cuda(0);
    Tensor result_bf16 = reduce_sum(gpu_bf16, {}, false).to_cpu();
    
    bfloat16_t bf16_result = result_bf16.data<bfloat16_t>()[0];
    std::cout << "BF16 sum: " << static_cast<float>(bf16_result) << "\n";
    std::cout << "Expected: ~6e10\n\n";
    
    // FP16 advantage: better precision for small numbers
    std::cout << "Testing small numbers (FP16 advantage):\n";
    Tensor fp16_tensor({{3}}, TensorOptions().with_dtype(Dtype::Float16));
    std::vector<float16_t> fp16_data = {
        float16_t(0.1234f),
        float16_t(0.5678f),
        float16_t(0.9012f)
    };
    fp16_tensor.set_data(fp16_data);
    
    Tensor gpu_fp16 = fp16_tensor.to_cuda(0);
    Tensor result_fp16 = reduce_sum(gpu_fp16, {}, false).to_cpu();
    
    float16_t fp16_result = result_fp16.data<float16_t>()[0];
    std::cout << "FP16 sum: " << static_cast<float>(fp16_result) << "\n";
    std::cout << "Expected: ~1.5924\n";
}
void test_integer_overflow_gpu() {
    print_test_header("GPU Integer Overflow Test");
    
    Tensor cpu_tensor({{100}}, TensorOptions().with_dtype(Dtype::Int16));
    std::vector<int16_t> data(100, 32000);  // 100 × 32000 = 3,200,000 (overflows int16)
    cpu_tensor.set_data(data);
    
    Tensor gpu_tensor = cpu_tensor.to_cuda(0);
    Tensor result = reduce_sum(gpu_tensor, {}, false).to_cpu();
    
    int64_t result_val = result.data<int64_t>()[0];
    std::cout << "Sum of 100 × 32000 (Int16→Int64): " << result_val << "\n";
    std::cout << "Expected: 3200000\n";
    
    if (result_val == 3200000) {
        std::cout << "✅ PASS: Integer widening prevented overflow\n";
    } else {
        std::cerr << "❌ FAIL: Expected 3200000, got " << result_val << "\n";
    }
}
int main() {
    std::cout << "========================================\n";
    std::cout << "  GPU REDUCTION OPERATIONS TEST SUITE\n";
    std::cout << "========================================\n\n";
    
    // Check CUDA availability
    if (!device::cuda_available()) {
        std::cerr << "ERROR: CUDA is not available!\n";
        return 1;
    }
    
    int device_count = device::cuda_device_count();
    std::cout << "CUDA devices found: " << device_count << "\n";
    std::cout << "Current device: " << device::get_current_cuda_device() << "\n\n";
    
    try {
        // Run all tests
        test_sum_reduction_gpu();
        test_mean_reduction_gpu();
        test_minmax_reduction_gpu();
        test_argminmax_reduction_gpu();
        test_product_reduction_gpu();
        test_nan_aware_reduction_gpu();
        test_keepdim_gpu();
        test_multi_axis_reduction_gpu();
        test_integer_reduction_gpu();
        test_fp16_reduction_gpu();
        test_large_tensor_reduction_gpu();
         test_fp16_numerical_accuracy();
    test_nan_aware_reductions_gpu();
    test_bf16_vs_fp16_comparison();
    test_integer_overflow_gpu();
        std::cout << "\n========================================\n";
        std::cout << "  ALL TESTS COMPLETED SUCCESSFULLY!\n";
        std::cout << "========================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}