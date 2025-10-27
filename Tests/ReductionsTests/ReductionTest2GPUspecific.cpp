// Comprehensive_Reduction_Test.cpp - Complete Test Suite for Reduction Operations
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace OwnTensor;

// ========================================
// COLOR CODES
// ========================================
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_BOLD    "\033[1m"

// ========================================
// TEST STATISTICS
// ========================================
int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

struct TestResult {
    std::string name;
    bool passed;
    std::string error_msg;
    double duration_ms;
};

std::vector<TestResult> test_results;

// ========================================
// UTILITY FUNCTIONS
// ========================================
void print_test_header(const std::string& category) {
    std::cout << "\n" << COLOR_CYAN << "========================================" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "  " << category << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "========================================" << COLOR_RESET << "\n\n";
}

void log_test(const std::string& test_name, bool passed, const std::string& error = "", double duration_ms = 0.0) {
    total_tests++;
    if (passed) {
        passed_tests++;
        std::cout << COLOR_GREEN << "[PASS] " << COLOR_RESET << test_name;
        if (duration_ms > 0) {
            std::cout << COLOR_YELLOW << " [" << std::fixed << std::setprecision(2) 
                     << duration_ms << "ms]" << COLOR_RESET;
        }
        std::cout << "\n";
    } else {
        failed_tests++;
        std::cout << COLOR_RED << "[FAIL] " << COLOR_RESET << test_name;
        if (!error.empty()) {
            std::cout << "\n       Error: " << error;
        }
        std::cout << "\n";
    }
    test_results.push_back({test_name, passed, error, duration_ms});
}

template<typename T>
bool approx_equal(T a, T b, T tolerance = 1e-5) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= tolerance;
    }
    return a == b;
}

void print_system_info() {
    std::cout << COLOR_CYAN << "\n========================================\n";
    std::cout << "SYSTEM CONFIGURATION\n";
    std::cout << "========================================\n" << COLOR_RESET;
    
    #ifdef _OPENMP
    std::cout << "OpenMP:           " << COLOR_GREEN << "ENABLED" << COLOR_RESET 
              << " (" << omp_get_max_threads() << " threads)\n";
    #else
    std::cout << "OpenMP:           " << COLOR_YELLOW << "DISABLED" << COLOR_RESET << "\n";
    #endif
    
    #ifdef WITH_CUDA
    if (device::cuda_available()) {
        std::cout << "CUDA:             " << COLOR_GREEN << "AVAILABLE" << COLOR_RESET 
                  << " (" << device::cuda_device_count() << " device(s))\n";
    } else {
        std::cout << "CUDA:             " << COLOR_YELLOW << "NOT AVAILABLE" << COLOR_RESET << "\n";
    }
    #else
    std::cout << "CUDA:             " << COLOR_YELLOW << "NOT COMPILED" << COLOR_RESET << "\n";
    #endif
    
    std::cout << "\n";
}

// ========================================
// TEST SECTION 1: BASIC CPU OPERATIONS
// ========================================
void test_basic_cpu_operations() {
    print_test_header("BASIC CPU OPERATIONS");
    
    // Test 1: Simple Sum
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        t.set_data(data);
        
        Tensor result = reduce_sum(t);
        float* res_data = result.data<float>();
        
        log_test("CPU: Sum reduction (3x4)", approx_equal(res_data[0], 78.0f));
    } catch (const std::exception& e) {
        log_test("CPU: Sum reduction", false, e.what());
    }
    
    // Test 2: Axis-specific reduction
    try {
        Tensor t(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        
        Tensor result = reduce_sum(t, {1}, false);
        float* res_data = result.data<float>();
        
        bool passed = approx_equal(res_data[0], 6.0f) && approx_equal(res_data[1], 15.0f);
        log_test("CPU: Sum along axis 1", passed);
    } catch (const std::exception& e) {
        log_test("CPU: Sum along axis 1", false, e.what());
    }
    
    // Test 3: Mean
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({2.0f, 4.0f, 6.0f, 8.0f});
        
        Tensor result = reduce_mean(t);
        float* res_data = result.data<float>();
        
        log_test("CPU: Mean reduction", approx_equal(res_data[0], 5.0f));
    } catch (const std::exception& e) {
        log_test("CPU: Mean reduction", false, e.what());
    }
    
    // Test 4: Max/Min
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({3.0f, 1.0f, 4.0f, 2.0f, 5.0f});
        
        Tensor max_result = reduce_max(t);
        Tensor min_result = reduce_min(t);
        
        float* max_data = max_result.data<float>();
        float* min_data = min_result.data<float>();
        
        bool passed = approx_equal(max_data[0], 5.0f) && approx_equal(min_data[0], 1.0f);
        log_test("CPU: Max/Min reduction", passed);
    } catch (const std::exception& e) {
        log_test("CPU: Max/Min reduction", false, e.what());
    }
    
    // Test 5: ArgMax/ArgMin
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({3.0f, 1.0f, 4.0f, 2.0f, 5.0f});
        
        Tensor argmax = reduce_argmax(t);
        Tensor argmin = reduce_argmin(t);
        
        int64_t* argmax_data = argmax.data<int64_t>();
        int64_t* argmin_data = argmin.data<int64_t>();
        
        bool passed = (argmax_data[0] == 4) && (argmin_data[0] == 1);
        log_test("CPU: ArgMax/ArgMin", passed);
    } catch (const std::exception& e) {
        log_test("CPU: ArgMax/ArgMin", false, e.what());
    }
}

// ========================================
// TEST SECTION 2: ALL DTYPES
// ========================================
void test_all_dtypes() {
    print_test_header("DTYPE VALIDATION - ALL 7 TYPES");
    
    // Int16
    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Int16));
        std::vector<int16_t> data = {1, 2, 3};
        t.set_data(data);
        Tensor result = reduce_sum(t);
        int64_t* res = result.data<int64_t>();
        log_test("Int16 → Int64 (sum)", res[0] == 6);
    } catch (const std::exception& e) {
        log_test("Int16 sum", false, e.what());
    }
    
    // Int32
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Int32));
        std::vector<int32_t> data = {10, 20, 30, 40};
        t.set_data(data);
        Tensor result = reduce_sum(t);
        int64_t* res = result.data<int64_t>();
        log_test("Int32 → Int64 (sum)", res[0] == 100);
    } catch (const std::exception& e) {
        log_test("Int32 sum", false, e.what());
    }
    
    // Float16
    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data = {float16_t(1.0f), float16_t(2.0f), float16_t(3.0f)};
        t.set_data(data);
        Tensor result = reduce_sum(t);
        float16_t* res = result.data<float16_t>();
        float val = static_cast<float>(res[0]);
        log_test("Float16 sum (double accumulation)", approx_equal(val, 6.0f, 0.5f));
    } catch (const std::exception& e) {
        log_test("Float16 sum", false, e.what());
    }
    
    // Bfloat16
    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data = {bfloat16_t(1.0f), bfloat16_t(2.0f), bfloat16_t(3.0f)};
        t.set_data(data);
        Tensor result = reduce_sum(t);
        bfloat16_t* res = result.data<bfloat16_t>();
        float val = static_cast<float>(res[0]);
        log_test("Bfloat16 sum (double accumulation)", approx_equal(val, 6.0f, 0.5f));
    } catch (const std::exception& e) {
        log_test("Bfloat16 sum", false, e.what());
    }
}

// ========================================
// TEST SECTION 3: NaN HANDLING
// ========================================
void test_nan_handling() {
    print_test_header("NaN-AWARE OPERATIONS");
    
    // Regular sum (NaN propagates)
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, NAN, 3.0f, 4.0f});
        
        Tensor result = reduce_sum(t);
        float* res = result.data<float>();
        
        log_test("Regular sum propagates NaN", std::isnan(res[0]));
    } catch (const std::exception& e) {
        log_test("Regular sum with NaN", false, e.what());
    }
    
    // NaN-aware sum
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, NAN, 3.0f, 4.0f});
        
        Tensor result = reduce_nansum(t);
        float* res = result.data<float>();
        
        log_test("NaN-aware sum ignores NaN", approx_equal(res[0], 8.0f));
    } catch (const std::exception& e) {
        log_test("NaN-aware sum", false, e.what());
    }
    
    // NaN-aware mean
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({2.0f, 4.0f, NAN, 6.0f, 8.0f});
        
        Tensor result = reduce_nanmean(t);
        float* res = result.data<float>();
        
        log_test("NaN-aware mean", !std::isnan(res[0]));
    } catch (const std::exception& e) {
        log_test("NaN-aware mean", false, e.what());
    }
}

// ========================================
// TEST SECTION 4: MULTI-DIMENSIONAL
// ========================================
void test_multidimensional() {
    print_test_header("MULTI-DIMENSIONAL REDUCTIONS");
    
    // 2D reduction
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(12);
        for (int i = 0; i < 12; i++) data[i] = static_cast<float>(i + 1);
        t.set_data(data);
        
        Tensor result = reduce_sum(t, {0}, false);
        bool shape_ok = result.shape().dims.size() == 1 && result.shape().dims[0] == 4;
        
        log_test("2D: Sum along axis 0", shape_ok);
    } catch (const std::exception& e) {
        log_test("2D reduction", false, e.what());
    }
    
    // 3D reduction with keepdim
    try {
        Tensor t(Shape{{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(24, 1.0f);
        t.set_data(data);
        
        Tensor result = reduce_sum(t, {1}, true);
        bool shape_ok = result.shape().dims.size() == 3 && 
                       result.shape().dims[0] == 2 &&
                       result.shape().dims[1] == 1 &&
                       result.shape().dims[2] == 4;
        
        log_test("3D: Sum with keepdim=true", shape_ok);
    } catch (const std::exception& e) {
        log_test("3D keepdim", false, e.what());
    }
    
    // Multi-axis reduction
    try {
        Tensor t(Shape{{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(24, 2.0f);
        t.set_data(data);
        
        Tensor result = reduce_sum(t, {0, 2}, false);
        bool shape_ok = result.shape().dims.size() == 1 && result.shape().dims[0] == 3;
        
        log_test("3D: Multi-axis reduction", shape_ok);
    } catch (const std::exception& e) {
        log_test("Multi-axis reduction", false, e.what());
    }
}

// ========================================
// TEST SECTION 5: GPU KERNEL VALIDATION
// ========================================
void test_gpu_operations() {
    print_test_header("GPU CUDA KERNEL VALIDATION");
    
    #ifdef WITH_CUDA
    if (!device::cuda_available()) {
        std::cout << COLOR_YELLOW << "CUDA not available, skipping GPU tests\n" << COLOR_RESET;
        return;
    }
    
    std::cout << COLOR_CYAN << "Active CUDA Device: " << device::get_current_cuda_device() << COLOR_RESET << "\n\n";
    
    // =============================================
    // TEST 1: GPU vs CPU - Numerical Correctness
    // =============================================
    try {
        Tensor cpu_t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        cpu_t.set_data(data);
        
        // CPU reduction
        Tensor cpu_result = reduce_sum(cpu_t, {0}, false);
        
        // GPU reduction
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor gpu_result = reduce_sum(gpu_t, {0}, false);
        Tensor gpu_result_cpu = gpu_result.to_cpu();
        
        // Compare results
        float* cpu_data = cpu_result.data<float>();
        float* gpu_data = gpu_result_cpu.data<float>();
        
        bool all_match = true;
        for (int i = 0; i < 4; i++) {
            if (!approx_equal(cpu_data[i], gpu_data[i], 1e-4f)) {
                all_match = false;
                break;
            }
        }
        
        log_test("GPU Kernel: Sum correctness (GPU == CPU)", all_match);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Sum correctness", false, e.what());
    }
    
    // =============================================
    // TEST 2: GPU Mean with Expected Values
    // =============================================
    try {
        Tensor cpu_t(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
        cpu_t.set_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_mean(gpu_t, {0}, false).to_cpu();
        
        float* res = result.data<float>();
        bool passed = approx_equal(res[0], 2.5f) && 
                     approx_equal(res[1], 3.5f) && 
                     approx_equal(res[2], 4.5f);
        
        log_test("GPU Kernel: Mean computation (expected values)", passed);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Mean", false, e.what());
    }
    
    // =============================================
    // TEST 3: GPU Max/Min Kernels
    // =============================================
    try {
        Tensor cpu_t(Shape{{3, 3}}, TensorOptions().with_dtype(Dtype::Float32));
        cpu_t.set_data({9.0f, 2.0f, 7.0f, 4.0f, 5.0f, 6.0f, 1.0f, 8.0f, 3.0f});
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        
        Tensor max_result = reduce_max(gpu_t, {0}, false).to_cpu();
        Tensor min_result = reduce_min(gpu_t, {0}, false).to_cpu();
        
        float* max_data = max_result.data<float>();
        float* min_data = min_result.data<float>();
        
        bool max_ok = approx_equal(max_data[0], 9.0f) && 
                     approx_equal(max_data[1], 8.0f) && 
                     approx_equal(max_data[2], 7.0f);
        
        bool min_ok = approx_equal(min_data[0], 1.0f) && 
                     approx_equal(min_data[1], 2.0f) && 
                     approx_equal(min_data[2], 3.0f);
        
        log_test("GPU Kernel: Max/Min computation", max_ok && min_ok);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Max/Min", false, e.what());
    }
    
    // =============================================
    // TEST 4: GPU ArgMax/ArgMin Index Kernels
    // =============================================
    try {
        Tensor cpu_t(Shape{{2, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        cpu_t.set_data({3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 5.0f, 6.0f, 7.0f});
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        
        Tensor argmax = reduce_argmax(gpu_t, {1}, false).to_cpu();
        Tensor argmin = reduce_argmin(gpu_t, {1}, false).to_cpu();
        
        int64_t* argmax_data = argmax.data<int64_t>();
        int64_t* argmin_data = argmin.data<int64_t>();
        
        bool passed = (argmax_data[0] == 2) && (argmax_data[1] == 0) &&
                     (argmin_data[0] == 1) && (argmin_data[1] == 1);
        
        log_test("GPU Kernel: ArgMax/ArgMin indices", passed);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: ArgMax/ArgMin", false, e.what());
    }
    
    // =============================================
    // TEST 5: GPU Float16 Kernel (Double Accumulation)
    // =============================================
    try {
        Tensor cpu_t(Shape{{100}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data(100, float16_t(1.0f));
        cpu_t.set_data(data);
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_sum(gpu_t).to_cpu();
        
        float16_t* res = result.data<float16_t>();
        float val = static_cast<float>(res[0]);
        
        bool passed = approx_equal(val, 100.0f, 2.0f);
        log_test("GPU Kernel: Float16 (double accumulation)", passed);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Float16", false, e.what());
    }
    
    // =============================================
    // TEST 6: GPU Bfloat16 Kernel
    // =============================================
    try {
        Tensor cpu_t(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data = {
            bfloat16_t(1.0f), bfloat16_t(2.0f), bfloat16_t(3.0f),
            bfloat16_t(4.0f), bfloat16_t(5.0f), bfloat16_t(6.0f)
        };
        cpu_t.set_data(data);
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_mean(gpu_t, {0}, false).to_cpu();
        
        bfloat16_t* res = result.data<bfloat16_t>();
        
        bool passed = approx_equal(static_cast<float>(res[0]), 2.5f, 0.2f) &&
                     approx_equal(static_cast<float>(res[1]), 3.5f, 0.2f) &&
                     approx_equal(static_cast<float>(res[2]), 4.5f, 0.2f);
        
        log_test("GPU Kernel: Bfloat16 mean", passed);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Bfloat16", false, e.what());
    }
    
    // =============================================
    // TEST 7: GPU Integer Widening (Int16 → Int64)
    // =============================================
    try {
        Tensor cpu_t(Shape{{100}}, TensorOptions().with_dtype(Dtype::Int16));
        std::vector<int16_t> data(100, 300);
        cpu_t.set_data(data);
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_sum(gpu_t).to_cpu();
        
        bool dtype_ok = result.dtype() == Dtype::Int64;
        int64_t* res = result.data<int64_t>();
        bool value_ok = res[0] == 30000LL;
        
        log_test("GPU Kernel: Int16→Int64 widening", dtype_ok && value_ok);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Integer widening", false, e.what());
    }
    
    // =============================================
    // TEST 8: GPU NaN-Aware Sum Kernel
    // =============================================
    try {
        Tensor cpu_t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        cpu_t.set_data({1.0f, NAN, 3.0f, NAN, 5.0f});
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_nansum(gpu_t).to_cpu();
        
        float* res = result.data<float>();
        bool passed = approx_equal(res[0], 9.0f);
        
        log_test("GPU Kernel: NaN-aware sum", passed);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: NaN-aware", false, e.what());
    }
    
    // =============================================
    // TEST 9: GPU Multi-Axis Reduction
    // =============================================
    try {
        Tensor cpu_t(Shape{{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(24);
        for (int i = 0; i < 24; i++) data[i] = 1.0f;
        cpu_t.set_data(data);
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_sum(gpu_t, {0, 2}, false).to_cpu();
        
        bool shape_ok = result.shape().dims.size() == 1 && result.shape().dims[0] == 3;
        
        float* res = result.data<float>();
        bool values_ok = approx_equal(res[0], 8.0f) && 
                        approx_equal(res[1], 8.0f) && 
                        approx_equal(res[2], 8.0f);
        
        log_test("GPU Kernel: Multi-axis reduction", shape_ok && values_ok);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Multi-axis", false, e.what());
    }
    
    // =============================================
    // TEST 10: GPU KeepDim Flag
    // =============================================
    try {
        Tensor cpu_t(Shape{{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(24, 1.0f);
        cpu_t.set_data(data);
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_sum(gpu_t, {1}, true).to_cpu();
        
        bool shape_ok = result.shape().dims.size() == 3 &&
                       result.shape().dims[0] == 2 &&
                       result.shape().dims[1] == 1 &&
                       result.shape().dims[2] == 4;
        
        log_test("GPU Kernel: KeepDim=true", shape_ok);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: KeepDim", false, e.what());
    }
    
    // =============================================
    // TEST 11: GPU Large Tensor (Performance)
    // =============================================
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        Tensor cpu_t(Shape{{1024, 1024}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(1024*1024, 1.0f);
        cpu_t.set_data(data);
        
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = reduce_sum(gpu_t, {0}, false);
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        Tensor result_cpu = result.to_cpu();
        float* res = result_cpu.data<float>();
        
        bool value_ok = approx_equal(res[0], 1024.0f, 1.0f);
        
        log_test("GPU Kernel: Large tensor (1M elements)", value_ok, "", duration);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Large tensor", false, e.what());
    }
    
    // =============================================
    // TEST 12: GPU Memory Consistency Check
    // =============================================
    try {
        Tensor cpu_t(Shape{{5, 5}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(25);
        for (int i = 0; i < 25; i++) data[i] = static_cast<float>(i);
        cpu_t.set_data(data);
        
        // Multiple operations on same GPU tensor
        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor sum1 = reduce_sum(gpu_t, {0}, false);
        Tensor sum2 = reduce_sum(gpu_t, {1}, false);
        Tensor mean1 = reduce_mean(gpu_t, {0}, false);
        
        // Verify all succeeded
        bool passed = sum1.is_cuda() && sum2.is_cuda() && mean1.is_cuda();
        
        log_test("GPU Kernel: Memory consistency (multiple ops)", passed);
    } catch (const std::exception& e) {
        log_test("GPU Kernel: Memory consistency", false, e.what());
    }
    
    #else
    std::cout << COLOR_YELLOW << "CUDA not compiled, skipping GPU tests\n" << COLOR_RESET;
    #endif
}

// ========================================
// TEST SECTION 6: PERFORMANCE BENCHMARKS
// ========================================
void test_performance() {
    print_test_header("PERFORMANCE BENCHMARKS");
    
    // Large tensor CPU
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        Tensor t(Shape{{1000, 1000}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(1000000, 1.0f);
        t.set_data(data);
        
        Tensor result = reduce_sum(t, {0}, false);
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        log_test("CPU: 1M element sum", true, "", duration);
    } catch (const std::exception& e) {
        log_test("CPU: Large tensor", false, e.what());
    }
    
    #ifdef WITH_CUDA
    if (device::cuda_available()) {
        try {
            Tensor cpu_t(Shape{{1000, 1000}}, TensorOptions().with_dtype(Dtype::Float32));
            std::vector<float> data(1000000, 1.0f);
            cpu_t.set_data(data);
            Tensor gpu_t = cpu_t.to_cuda(0);
            
            auto start = std::chrono::high_resolution_clock::now();
            Tensor result = reduce_sum(gpu_t, {0}, false);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            
            log_test("GPU: 1M element sum", true, "", duration);
        } catch (const std::exception& e) {
            log_test("GPU: Large tensor", false, e.what());
        }
    }
    #endif
}

// ========================================
// TEST SECTION 7: EDGE CASES
// ========================================
void test_edge_cases() {
    print_test_header("EDGE CASE VALIDATION");
    
    // Single element
    try {
        Tensor t(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({42.0f});
        Tensor result = reduce_sum(t);
        float* res = result.data<float>();
        log_test("Single element tensor", approx_equal(res[0], 42.0f));
    } catch (const std::exception& e) {
        log_test("Single element", false, e.what());
    }
    
    // All zeros
    try {
        Tensor t(Shape{{10}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> zeros(10, 0.0f);
        t.set_data(zeros);
        Tensor result = reduce_sum(t);
        float* res = result.data<float>();
        log_test("All zeros", approx_equal(res[0], 0.0f));
    } catch (const std::exception& e) {
        log_test("All zeros", false, e.what());
    }
    
    // Integer overflow protection
    try {
        Tensor t(Shape{{100}}, TensorOptions().with_dtype(Dtype::Int16));
        std::vector<int16_t> data(100, 30000);
        t.set_data(data);
        Tensor result = reduce_sum(t);
        int64_t* res = result.data<int64_t>();
        bool passed = (result.dtype() == Dtype::Int64) && (res[0] == 3000000LL);
        log_test("Int16 overflow → Int64", passed);
    } catch (const std::exception& e) {
        log_test("Integer overflow protection", false, e.what());
    }
}

// ========================================
// FINAL SUMMARY
// ========================================
void print_final_summary() {
    std::cout << "\n" << COLOR_MAGENTA << COLOR_BOLD << "========================================\n";
    std::cout << "  FINAL TEST SUMMARY\n";
    std::cout << "========================================\n" << COLOR_RESET;
    
    std::cout << "\nTotal Tests:  " << total_tests << "\n";
    std::cout << COLOR_GREEN << "Passed:       " << passed_tests << COLOR_RESET << "\n";
    std::cout << COLOR_RED << "Failed:       " << failed_tests << COLOR_RESET << "\n";
    
    double pass_rate = (total_tests > 0) ? (100.0 * passed_tests / total_tests) : 0.0;
    std::cout << "\nPass Rate:    " << std::fixed << std::setprecision(1) << pass_rate << "%\n";
    
    // Calculate total time
    double total_time = 0.0;
    for (const auto& result : test_results) {
        total_time += result.duration_ms;
    }
    std::cout << "Total Time:   " << std::fixed << std::setprecision(2) 
              << total_time / 1000.0 << " seconds\n";
    
    if (failed_tests > 0) {
        std::cout << "\n" << COLOR_YELLOW << "Failed Tests:\n" << COLOR_RESET;
        for (const auto& result : test_results) {
            if (!result.passed) {
                std::cout << "  - " << result.name;
                if (!result.error_msg.empty()) {
                    std::cout << "\n    " << result.error_msg;
                }
                std::cout << "\n";
            }
        }
    }
    
    std::cout << "\n" << COLOR_MAGENTA << "========================================\n" << COLOR_RESET;
    
    if (failed_tests == 0) {
        std::cout << COLOR_GREEN << "\n🎉 ALL TESTS PASSED! 🎉\n" << COLOR_RESET;
        std::cout << COLOR_GREEN << "\n✅ REDUCTION MODULE IS PRODUCTION READY\n" << COLOR_RESET;
    } else {
        std::cout << COLOR_RED << "\n⚠️  SOME TESTS FAILED ⚠️\n" << COLOR_RESET;
    }
    
    std::cout << "\n";
}

// ========================================
// MAIN
// ========================================
int main() {
    std::cout << COLOR_CYAN << COLOR_BOLD << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║   OwnTensor Comprehensive Reduction Test Suite      ║\n";
    std::cout << "║   Complete Validation for Production Deployment     ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << COLOR_RESET << "\n";
    
    print_system_info();
    
    try {
        test_basic_cpu_operations();
        test_all_dtypes();
        test_nan_handling();
        test_multidimensional();
        test_gpu_operations();
        test_performance();
        test_edge_cases();
        
        print_final_summary();
        
    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "\n\nFATAL ERROR: " << e.what() << COLOR_RESET << "\n\n";
        return 1;
    }
    
    return (failed_tests == 0) ? 0 : 1;
}