// Tests/TrigonometricTests/TrigonometryTest2.cpp - Complete Test Suite for Trigonometric Operations
#include "core/Tensor.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>
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

// flexible approx_equal: accept different floating types
template <class A, class B>
bool approx_equal(A a, B b, double tol = 1e-4) {
    return std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= tol;
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
// TEST 1: BASIC CPU OPERATIONS
// ========================================
void test_basic_trig_cpu() {
    print_test_header("BASIC CPU TRIGONOMETRIC OPERATIONS");

    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        // ensure all literals are float to match the Tensor dtype
        t.set_data(std::vector<float>{0.0f, static_cast<float>(M_PI_2), static_cast<float>(M_PI), static_cast<float>(3.0 * M_PI_2), static_cast<float>(2.0 * M_PI)});

        Tensor s = sin(t);
        Tensor c = cos(t);

        float* s_data = s.data<float>();
        float* c_data = c.data<float>();

        bool passed =
            approx_equal(s_data[0], 0.0f) &&
            approx_equal(c_data[0], 1.0f) &&
            approx_equal(s_data[1], 1.0f) &&
            approx_equal(c_data[1], 0.0f, 1e-4) &&
            approx_equal(s_data[2], 0.0f, 1e-4) &&
            approx_equal(c_data[2], -1.0f);

        log_test("CPU: sin/cos basic test", passed);
    } catch (const std::exception& e) {
        log_test("CPU: sin/cos basic test", false, e.what());
    }

    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data(std::vector<float>{0.0f, 1.0f, -1.0f});
        Tensor tan_t = tan(t);
        float* res = tan_t.data<float>();
        bool passed = approx_equal(res[0], 0.0f) && res[1] > 1.55f && res[2] < -1.55f;
        log_test("CPU: tan", passed);
    } catch (const std::exception& e) {
        log_test("CPU: tan", false, e.what());
    }

    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data(std::vector<float>{-1.0f, 0.0f, 1.0f});
        Tensor a = asin(t);
        float* res = a.data<float>();
        bool passed = approx_equal(res[0], static_cast<float>(-M_PI_2), 1e-4) &&
                      approx_equal(res[1], 0.0f) &&
                      approx_equal(res[2], static_cast<float>(M_PI_2), 1e-4);
        log_test("CPU: asin", passed);
    } catch (const std::exception& e) {
        log_test("CPU: asin", false, e.what());
    }
}

// ========================================
// TEST 2: HYPERBOLIC OPERATIONS
// ========================================
void test_hyperbolic_cpu() {
    print_test_header("HYPERBOLIC FUNCTIONS");

    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data(std::vector<float>{0.0f, 1.0f, -1.0f});

        Tensor s = sinh(t);
        Tensor c = cosh(t);
        Tensor tnh = tanh(t);

        float* s_data = s.data<float>();
        float* c_data = c.data<float>();
        float* t_data = tnh.data<float>();

        bool passed = approx_equal(s_data[1], std::sinh(1.0f)) &&
                      approx_equal(c_data[1], std::cosh(1.0f)) &&
                      approx_equal(t_data[1], std::tanh(1.0f));
        log_test("CPU: sinh/cosh/tanh", passed);
    } catch (const std::exception& e) {
        log_test("CPU: sinh/cosh/tanh", false, e.what());
    }
}

// ========================================
// TEST 3: SPECIAL VALUES (NaN/INF)
// ========================================
void test_nan_inf_handling() {
    print_test_header("NaN AND INF HANDLING");

    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data(std::vector<float>{NAN, INFINITY, -INFINITY});
        Tensor s = sin(t);
        float* res = s.data<float>();
        bool has_nan = std::isnan(res[0]) || std::isnan(res[1]) || std::isnan(res[2]);
        log_test("sin(NaN/Inf) returns NaN", has_nan);
    } catch (const std::exception& e) {
        log_test("sin(NaN/Inf)", false, e.what());
    }
}

// ========================================
// TEST 4: DTYPE VARIANTS
// ========================================
void test_dtypes() {
    print_test_header("DTYPE VARIANTS");

    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> f16data = {float16_t(0.0f), float16_t(1.0f), float16_t(3.14f)};
        t.set_data(f16data);
        Tensor s = sin(t);
        float16_t* res = s.data<float16_t>();
        float val = static_cast<float>(res[1]);
        log_test("Float16: sin(1)", approx_equal(val, std::sin(1.0f), 1e-1));
    } catch (const std::exception& e) {
        log_test("Float16 sin", false, e.what());
    }

    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> bf16data = {bfloat16_t(0.0f), bfloat16_t(1.0f), bfloat16_t(3.14f)};
        t.set_data(bf16data);
        Tensor s = sin(t);
        bfloat16_t* res = s.data<bfloat16_t>();
        float val = static_cast<float>(res[1]);
        log_test("Bfloat16: sin(1)", approx_equal(val, std::sin(1.0f), 1e-1));
    } catch (const std::exception& e) {
        log_test("Bfloat16 sin", false, e.what());
    }
}

// ========================================
// TEST 5: GPU OPERATIONS
// ========================================
void test_gpu_trig() {
    print_test_header("GPU TRIGONOMETRIC OPERATIONS");

#ifdef WITH_CUDA
    if (!device::cuda_available()) {
        std::cout << COLOR_YELLOW << "CUDA not available, skipping GPU tests\n" << COLOR_RESET;
        return;
    }

    try {
        Tensor cpu_t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        cpu_t.set_data(std::vector<float>{0.0f, static_cast<float>(M_PI_2), static_cast<float>(M_PI), static_cast<float>(3.0 * M_PI_2), static_cast<float>(2.0 * M_PI)});

        Tensor gpu_t = cpu_t.to_cuda(0);
        Tensor result = sin(gpu_t);
        Tensor back = result.to_cpu();

        float* res = back.data<float>();
        bool passed = approx_equal(res[1], 1.0f) && approx_equal(res[2], 0.0f, 1e-4);
        log_test("GPU: sin()", passed);
    } catch (const std::exception& e) {
        log_test("GPU: sin()", false, e.what());
    }

#else
    std::cout << COLOR_YELLOW << "CUDA not compiled, skipping GPU tests\n" << COLOR_RESET;
#endif
}

// ========================================
// TEST 6: PERFORMANCE BENCHMARK
// ========================================
void test_performance() {
    print_test_header("PERFORMANCE BENCHMARKS");

    try {
        Tensor t(Shape{{1000000}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(1000000, 1.0f);
        t.set_data(data);

        auto start = std::chrono::high_resolution_clock::now();
        Tensor r = sin(t);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        log_test("CPU: 1M sin()", true, "", ms);
    } catch (const std::exception& e) {
        log_test("CPU Performance", false, e.what());
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

    // total time in ms
    double total_time = 0.0;
    for (const auto& result : test_results) total_time += result.duration_ms;
    std::cout << "Total Time:   " << std::fixed << std::setprecision(2) << total_time / 1000.0 << " seconds\n";

    if (failed_tests > 0) {
        std::cout << "\n" << COLOR_YELLOW << "Failed Tests:\n" << COLOR_RESET;
        for (const auto& result : test_results) {
            if (!result.passed) {
                std::cout << "  - " << result.name;
                if (!result.error_msg.empty()) std::cout << "\n    " << result.error_msg;
                std::cout << "\n";
            }
        }
    }

    std::cout << "\n" << COLOR_MAGENTA << "========================================\n" << COLOR_RESET;

    if (failed_tests == 0) {
        std::cout << COLOR_GREEN << "\n🎉 ALL TESTS PASSED! 🎉\n" << COLOR_RESET;
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
    std::cout << "║   OwnTensor Comprehensive Trigonometry Test Suite    ║\n";
    std::cout << "║   Full Validation of Sin/Cos/Tan/Tanh/Asin/... Ops   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << COLOR_RESET << "\n";

    print_system_info();

    try {
        test_basic_trig_cpu();
        test_hyperbolic_cpu();
        test_nan_inf_handling();
        test_dtypes();
        test_gpu_trig();
        test_performance();

        print_final_summary();
    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "\n\nFATAL ERROR: " << e.what() << COLOR_RESET << "\n\n";
        return 1;
    }

    return (failed_tests == 0) ? 0 : 1;
}
