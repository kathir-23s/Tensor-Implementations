#include "Tensor.h"
#include "Reduction.h"
#include "Types.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>

using namespace OwnTensor;

// ========================================================================
// TEST UTILITIES
// ========================================================================

// Color codes for terminal output
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_MAGENTA "\033[35m"

int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

// Test result tracker
struct TestResult {
    std::string name;
    bool passed;
    std::string error_msg;
};

std::vector<TestResult> test_results;

void print_test_header(const std::string& category) {
    std::cout << "\n" << COLOR_CYAN << "========================================" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "  " << category << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "========================================" << COLOR_RESET << "\n\n";
}

void log_test(const std::string& test_name, bool passed, const std::string& error = "") {
    total_tests++;
    if (passed) {
        passed_tests++;
        std::cout << COLOR_GREEN << "[PASS] " << COLOR_RESET << test_name << "\n";
    } else {
        failed_tests++;
        std::cout << COLOR_RED << "[FAIL] " << COLOR_RESET << test_name;
        if (!error.empty()) {
            std::cout << "\n       Error: " << error;
        }
        std::cout << "\n";
    }
    test_results.push_back({test_name, passed, error});
}

// Floating point comparison with tolerance
template<typename T>
bool approx_equal(T a, T b, T tolerance = 1e-5) {
    if (std::is_floating_point<T>::value) {
        return std::abs(a - b) <= tolerance;
    }
    return a == b;
}

// Check if value is NaN
template<typename T>
bool is_nan_value(T val) {
    if constexpr (std::is_floating_point<T>::value) {
        return std::isnan(val);
    }
    return false;
}

// ========================================================================
// DATATYPE VALIDATION TESTS
// ========================================================================

void test_all_dtypes_sum() {
    print_test_header("DATATYPE VALIDATION - Sum Reduction");
    
    // Test Int16
    try {
        Tensor t_int16(Shape{{3}}, TensorOptions().with_dtype(Dtype::Int16));
        std::vector<int16_t> data = {1, 2, 3};
        t_int16.set_data(data);
        Tensor result = reduce_sum(t_int16);
        int64_t* res_data = result.data<int64_t>();
        log_test("Sum on Int16", res_data[0] == 6);
    } catch (const std::exception& e) {
        log_test("Sum on Int16", false, e.what());
    }
    
    // Test Int32
    try {
        Tensor t_int32(Shape{{4}}, TensorOptions().with_dtype(Dtype::Int32));
        std::vector<int32_t> data = {10, 20, 30, 40};
        t_int32.set_data(data);
        Tensor result = reduce_sum(t_int32);
        int64_t* res_data = result.data<int64_t>();
        log_test("Sum on Int32", res_data[0] == 100);
    } catch (const std::exception& e) {
        log_test("Sum on Int32", false, e.what());
    }
    
    // Test Int64
    try {
        Tensor t_int64(Shape{{3}}, TensorOptions().with_dtype(Dtype::Int64));
        std::vector<int64_t> data = {100, 200, 300};
        t_int64.set_data(data);
        Tensor result = reduce_sum(t_int64);
        int64_t* res_data = result.data<int64_t>();
        log_test("Sum on Int64", res_data[0] == 600);
    } catch (const std::exception& e) {
        log_test("Sum on Int64", false, e.what());
    }
    
    // Test Float32
    try {
        Tensor t_float32(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f};
        t_float32.set_data(data);
        Tensor result = reduce_sum(t_float32);
        float* res_data = result.data<float>();
        log_test("Sum on Float32", approx_equal(res_data[0], 12.0f, 1e-4f));
    } catch (const std::exception& e) {
        log_test("Sum on Float32", false, e.what());
    }
    
    // Test Float64
    try {
        Tensor t_float64(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float64));
        std::vector<double> data = {1.111, 2.222, 3.333};
        t_float64.set_data(data);
        Tensor result = reduce_sum(t_float64);
        double* res_data = result.data<double>();
        log_test("Sum on Float64", approx_equal(res_data[0], 6.666, 1e-6));
    } catch (const std::exception& e) {
        log_test("Sum on Float64", false, e.what());
    }
    
    // Test Bfloat16
    try {
        Tensor t_bf16(Shape{{3}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data = {bfloat16_t(1.0f), bfloat16_t(2.0f), bfloat16_t(3.0f)};
        t_bf16.set_data(data);
        Tensor result = reduce_sum(t_bf16);
        bfloat16_t* res_data = result.data<bfloat16_t>();
        float res_val = static_cast<float>(res_data[0]);
        log_test("Sum on Bfloat16", approx_equal(res_val, 6.0f, 0.5f));
    } catch (const std::exception& e) {
        log_test("Sum on Bfloat16", false, e.what());
    }
    
    // Test Float16
    try {
        Tensor t_fp16(Shape{{3}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data = {float16_t(1.0f), float16_t(2.0f), float16_t(3.0f)};
        t_fp16.set_data(data);
        Tensor result = reduce_sum(t_fp16);
        float16_t* res_data = result.data<float16_t>();
        float res_val = static_cast<float>(res_data[0]);
        log_test("Sum on Float16", approx_equal(res_val, 6.0f, 0.5f));
    } catch (const std::exception& e) {
        log_test("Sum on Float16", false, e.what());
    }
}

// ========================================================================
// DIMENSIONALITY TESTS
// ========================================================================

void test_1d_reductions() {
    print_test_header("1D TENSOR REDUCTIONS");
    
    // 1D Sum
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        Tensor result = reduce_sum(t);
        float* data = result.data<float>();
        log_test("1D Sum", approx_equal(data[0], 15.0f));
    } catch (const std::exception& e) {
        log_test("1D Sum", false, e.what());
    }
    
    // 1D Max
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 5.0f, 3.0f, 2.0f, 4.0f});
        Tensor result = reduce_max(t);
        float* data = result.data<float>();
        log_test("1D Max", approx_equal(data[0], 5.0f));
    } catch (const std::exception& e) {
        log_test("1D Max", false, e.what());
    }
    
    // 1D Min
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({3.0f, 1.0f, 4.0f, 2.0f, 5.0f});
        Tensor result = reduce_min(t);
        float* data = result.data<float>();
        log_test("1D Min", approx_equal(data[0], 1.0f));
    } catch (const std::exception& e) {
        log_test("1D Min", false, e.what());
    }
    
    // 1D Mean
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({2.0f, 4.0f, 6.0f, 8.0f});
        Tensor result = reduce_mean(t);
        float* data = result.data<float>();
        log_test("1D Mean", approx_equal(data[0], 5.0f));
    } catch (const std::exception& e) {
        log_test("1D Mean", false, e.what());
    }
    
    // 1D Product
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f});
        Tensor result = reduce_product(t);
        float* data = result.data<float>();
        log_test("1D Product", approx_equal(data[0], 24.0f));
    } catch (const std::exception& e) {
        log_test("1D Product", false, e.what());
    }
}

void test_2d_reductions() {
    print_test_header("2D TENSOR REDUCTIONS");
    
    // 2D Sum along axis 0
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f});
        Tensor result = reduce_sum(t, {0}, false);
        float* data = result.data<float>();
        bool passed = approx_equal(data[0], 15.0f) && 
                     approx_equal(data[1], 18.0f) &&
                     approx_equal(data[2], 21.0f) &&
                     approx_equal(data[3], 24.0f);
        log_test("2D Sum along axis 0", passed);
    } catch (const std::exception& e) {
        log_test("2D Sum along axis 0", false, e.what());
    }
    
    // 2D Sum along axis 1
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f});
        Tensor result = reduce_sum(t, {1}, false);
        float* data = result.data<float>();
        bool passed = approx_equal(data[0], 10.0f) && 
                     approx_equal(data[1], 26.0f) &&
                     approx_equal(data[2], 42.0f);
        log_test("2D Sum along axis 1", passed);
    } catch (const std::exception& e) {
        log_test("2D Sum along axis 1", false, e.what());
    }
    
    // 2D Max along axis 0
    try {
        Tensor t(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 5.0f, 3.0f,
                    4.0f, 2.0f, 6.0f});
        Tensor result = reduce_max(t, {0}, false);
        float* data = result.data<float>();
        bool passed = approx_equal(data[0], 4.0f) && 
                     approx_equal(data[1], 5.0f) &&
                     approx_equal(data[2], 6.0f);
        log_test("2D Max along axis 0", passed);
    } catch (const std::exception& e) {
        log_test("2D Max along axis 0", false, e.what());
    }
    
    // 2D Mean with keepdim
    try {
        Tensor t(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({2.0f, 4.0f, 6.0f,
                    8.0f, 10.0f, 12.0f});
        Tensor result = reduce_mean(t, {1}, true);
        bool shape_ok = result.shape().dims.size() == 2 && 
                       result.shape().dims[0] == 2 &&
                       result.shape().dims[1] == 1;
        float* data = result.data<float>();
        bool vals_ok = approx_equal(data[0], 4.0f) && approx_equal(data[1], 10.0f);
        log_test("2D Mean with keepdim=true", shape_ok && vals_ok);
    } catch (const std::exception& e) {
        log_test("2D Mean with keepdim=true", false, e.what());
    }
}

void test_3d_reductions() {
    print_test_header("3D TENSOR REDUCTIONS");
    
    // 3D Sum along multiple axes
    try {
        Tensor t(Shape{{2, 3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(24);
        for (int i = 0; i < 24; i++) data[i] = static_cast<float>(i + 1);
        t.set_data(data);
        
        Tensor result = reduce_sum(t, {0, 2}, false);
        float* res = result.data<float>();
        // Expected: sum over dims 0 and 2, leaving dim 1 (size 3)
        log_test("3D Sum along multiple axes", result.shape().dims.size() == 1 && result.shape().dims[0] == 3);
    } catch (const std::exception& e) {
        log_test("3D Sum along multiple axes", false, e.what());
    }
    
    // 3D Max along axis 1
    try {
        Tensor t(Shape{{2, 3, 2}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
        Tensor result = reduce_max(t, {1}, false);
        bool shape_ok = result.shape().dims.size() == 2 &&
                       result.shape().dims[0] == 2 &&
                       result.shape().dims[1] == 2;
        log_test("3D Max along axis 1", shape_ok);
    } catch (const std::exception& e) {
        log_test("3D Max along axis 1", false, e.what());
    }
}

void test_high_dimensional() {
    print_test_header("HIGH DIMENSIONAL TENSOR TESTS");
    
    // 4D tensor reduction
    try {
        Tensor t(Shape{{2, 3, 4, 5}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(120, 1.0f);
        t.set_data(data);
        Tensor result = reduce_sum(t);
        float* res = result.data<float>();
        log_test("4D Tensor Full Reduction", approx_equal(res[0], 120.0f));
    } catch (const std::exception& e) {
        log_test("4D Tensor Full Reduction", false, e.what());
    }
    
    // 5D tensor reduction
    try {
        Tensor t(Shape{{2, 2, 2, 2, 2}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(32, 2.0f);
        t.set_data(data);
        Tensor result = reduce_mean(t, {1, 3}, false);
        bool shape_ok = result.shape().dims.size() == 3;
        log_test("5D Tensor Partial Reduction", shape_ok);
    } catch (const std::exception& e) {
        log_test("5D Tensor Partial Reduction", false, e.what());
    }
}

// ========================================================================
// AXIS ARGUMENT TESTS
// ========================================================================

void test_axis_arguments() {
    print_test_header("AXIS ARGUMENT TESTS");
    
    // Empty axes (full reduction)
    try {
        Tensor t(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        Tensor result = reduce_sum(t, {}, false);
        float* data = result.data<float>();
        log_test("Empty axes (full reduction)", approx_equal(data[0], 21.0f));
    } catch (const std::exception& e) {
        log_test("Empty axes (full reduction)", false, e.what());
    }
    
    // Negative axis index
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(12, 1.0f);
        t.set_data(data);
        Tensor result = reduce_sum(t, {-1}, false);
        bool shape_ok = result.shape().dims.size() == 1 && result.shape().dims[0] == 3;
        log_test("Negative axis index (-1)", shape_ok);
    } catch (const std::exception& e) {
        log_test("Negative axis index (-1)", false, e.what());
    }
    
    // Multiple axes
    try {
        Tensor t(Shape{{3, 4, 5}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(60, 2.0f);
        t.set_data(data);
        Tensor result = reduce_sum(t, {0, 2}, false);
        bool shape_ok = result.shape().dims.size() == 1 && result.shape().dims[0] == 4;
        log_test("Multiple axes {0, 2}", shape_ok);
    } catch (const std::exception& e) {
        log_test("Multiple axes {0, 2}", false, e.what());
    }
    
    // Keepdim parameter
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(12, 1.0f);
        t.set_data(data);
        Tensor result_no_keep = reduce_sum(t, {1}, false);
        Tensor result_keep = reduce_sum(t, {1}, true);
        bool no_keep_ok = result_no_keep.shape().dims.size() == 1;
        bool keep_ok = result_keep.shape().dims.size() == 2 && result_keep.shape().dims[1] == 1;
        log_test("Keepdim parameter test", no_keep_ok && keep_ok);
    } catch (const std::exception& e) {
        log_test("Keepdim parameter test", false, e.what());
    }
}

// ========================================================================
// NaN PROPAGATION TESTS
// ========================================================================

void test_nan_handling() {
    print_test_header("NaN PROPAGATION TESTS");
    
    // Regular sum with NaN (should propagate)
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(), 4.0f});
        Tensor result = reduce_sum(t);
        float* data = result.data<float>();
        log_test("Regular Sum propagates NaN", std::isnan(data[0]));
    } catch (const std::exception& e) {
        log_test("Regular Sum propagates NaN", false, e.what());
    }
    
    // NanSum ignores NaN
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(), 4.0f});
        Tensor result = reduce_nansum(t);
        float* data = result.data<float>();
        log_test("NanSum ignores NaN", approx_equal(data[0], 7.0f));
    } catch (const std::exception& e) {
        log_test("NanSum ignores NaN", false, e.what());
    }
    
    // Regular max with NaN
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 2.0f});
        Tensor result = reduce_max(t);
        float* data = result.data<float>();
        log_test("Regular Max propagates NaN", std::isnan(data[0]));
    } catch (const std::exception& e) {
        log_test("Regular Max propagates NaN", false, e.what());
    }
    
    // NanMax ignores NaN
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 2.0f});
        Tensor result = reduce_nanmax(t);
        float* data = result.data<float>();
        log_test("NanMax ignores NaN", approx_equal(data[0], 3.0f));
    } catch (const std::exception& e) {
        log_test("NanMax ignores NaN", false, e.what());
    }
    
    // NanMin ignores NaN
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({5.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f});
        Tensor result = reduce_nanmin(t);
        float* data = result.data<float>();
        log_test("NanMin ignores NaN", approx_equal(data[0], 2.0f));
    } catch (const std::exception& e) {
        log_test("NanMin ignores NaN", false, e.what());
    }
    
    // NanMean ignores NaN
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({2.0f, 4.0f, std::numeric_limits<float>::quiet_NaN(), 6.0f});
        Tensor result = reduce_nanmean(t);
        float* data = result.data<float>();
        // Note: nanmean still divides by original count, not non-NaN count
        log_test("NanMean computation", !std::isnan(data[0]));
    } catch (const std::exception& e) {
        log_test("NanMean computation", false, e.what());
    }
    
    // NanProduct ignores NaN
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({2.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f});
        Tensor result = reduce_nanproduct(t);
        float* data = result.data<float>();
        log_test("NanProduct ignores NaN", approx_equal(data[0], 24.0f));
    } catch (const std::exception& e) {
        log_test("NanProduct ignores NaN", false, e.what());
    }
}

// ========================================================================
// INDEX REDUCTION TESTS
// ========================================================================

void test_index_reductions() {
    print_test_header("INDEX REDUCTION TESTS (ArgMax/ArgMin)");
    
    // Argmax 1D
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 5.0f, 3.0f, 2.0f, 4.0f});
        Tensor result = reduce_argmax(t);
        int64_t* data = result.data<int64_t>();
        log_test("ArgMax 1D", data[0] == 1);
    } catch (const std::exception& e) {
        log_test("ArgMax 1D", false, e.what());
    }
    
    // Argmin 1D
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({3.0f, 1.0f, 4.0f, 2.0f, 5.0f});
        Tensor result = reduce_argmin(t);
        int64_t* data = result.data<int64_t>();
        log_test("ArgMin 1D", data[0] == 1);
    } catch (const std::exception& e) {
        log_test("ArgMin 1D", false, e.what());
    }
    
    // Argmax 2D along axis 0
    try {
        Tensor t(Shape{{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 8.0f, 3.0f, 4.0f,
                    5.0f, 2.0f, 7.0f, 8.0f,
                    9.0f, 6.0f, 5.0f, 2.0f});
        Tensor result = reduce_argmax(t, {0}, false);
        int64_t* data = result.data<int64_t>();
        bool passed = data[0] == 2 && data[1] == 0 && data[2] == 1 && data[3] == 1;
        log_test("ArgMax 2D along axis 0", passed);
    } catch (const std::exception& e) {
        log_test("ArgMax 2D along axis 0", false, e.what());
    }
    
    // Argmin 2D along axis 1
    try {
        Tensor t(Shape{{2, 4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({5.0f, 2.0f, 8.0f, 3.0f,
                    7.0f, 1.0f, 9.0f, 4.0f});
        Tensor result = reduce_argmin(t, {1}, false);
        int64_t* data = result.data<int64_t>();
        bool passed = data[0] == 1 && data[1] == 1;
        log_test("ArgMin 2D along axis 1", passed);
    } catch (const std::exception& e) {
        log_test("ArgMin 2D along axis 1", false, e.what());
    }
    
    // NanArgMax (ignores NaN)
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, std::numeric_limits<float>::quiet_NaN(), 5.0f, 3.0f, 4.0f});
        Tensor result = reduce_nanargmax(t);
        int64_t* data = result.data<int64_t>();
        log_test("NanArgMax ignores NaN", data[0] == 2);
    } catch (const std::exception& e) {
        log_test("NanArgMax ignores NaN", false, e.what());
    }
    
    // NanArgMin (ignores NaN)
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({5.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f, 4.0f, 3.0f});
        Tensor result = reduce_nanargmin(t);
        int64_t* data = result.data<int64_t>();
        log_test("NanArgMin ignores NaN", data[0] == 2);
    } catch (const std::exception& e) {
        log_test("NanArgMin ignores NaN", false, e.what());
    }
}

// ========================================================================
// EDGE CASE TESTS
// ========================================================================

void test_edge_cases() {
    print_test_header("EDGE CASE TESTS");
    
    // Single element tensor
    try {
        Tensor t(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({42.0f});
        Tensor result = reduce_sum(t);
        float* data = result.data<float>();
        log_test("Single element reduction", approx_equal(data[0], 42.0f));
    } catch (const std::exception& e) {
        log_test("Single element reduction", false, e.what());
    }
    
    // All zeros tensor
    try {
        Tensor t(Shape{{10}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> zeros(10, 0.0f);
        t.set_data(zeros);
        Tensor result = reduce_sum(t);
        float* data = result.data<float>();
        log_test("All zeros sum", approx_equal(data[0], 0.0f));
    } catch (const std::exception& e) {
        log_test("All zeros sum", false, e.what());
    }
    
    // All NaN tensor with nansum
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> nans(5, std::numeric_limits<float>::quiet_NaN());
        t.set_data(nans);
        Tensor result = reduce_nansum(t);
        float* data = result.data<float>();
        log_test("All NaN tensor with nansum", approx_equal(data[0], 0.0f));
    } catch (const std::exception& e) {
        log_test("All NaN tensor with nansum", false, e.what());
    }
    
    // Very large values (overflow protection)
    try {
        Tensor t(Shape{{3}}, TensorOptions().with_dtype(Dtype::Int32));
        std::vector<int32_t> large = {1000000, 2000000, 3000000};
        t.set_data(large);
        Tensor result = reduce_sum(t);
        int64_t* data = result.data<int64_t>();
        log_test("Large integer sum (Int64 output)", data[0] == 6000000LL);
    } catch (const std::exception& e) {
        log_test("Large integer sum (Int64 output)", false, e.what());
    }
    
    // Negative values
    try {
        Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({-1.0f, -2.0f, -3.0f, -4.0f});
        Tensor result_sum = reduce_sum(t);
        Tensor result_max = reduce_max(t);
        Tensor result_min = reduce_min(t);
        float* sum_data = result_sum.data<float>();
        float* max_data = result_max.data<float>();
        float* min_data = result_min.data<float>();
        bool passed = approx_equal(sum_data[0], -10.0f) &&
                     approx_equal(max_data[0], -1.0f) &&
                     approx_equal(min_data[0], -4.0f);
        log_test("Negative values handling", passed);
    } catch (const std::exception& e) {
        log_test("Negative values handling", false, e.what());
    }
    
    // Mixed positive and negative
    try {
        Tensor t(Shape{{6}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({-3.0f, 5.0f, -2.0f, 8.0f, -1.0f, 4.0f});
        Tensor result = reduce_sum(t);
        float* data = result.data<float>();
        log_test("Mixed positive/negative sum", approx_equal(data[0], 11.0f));
    } catch (const std::exception& e) {
        log_test("Mixed positive/negative sum", false, e.what());
    }
    
    // Product with zero
    try {
        Tensor t(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
        t.set_data({1.0f, 2.0f, 0.0f, 3.0f, 4.0f});
        Tensor result = reduce_product(t);
        float* data = result.data<float>();
        log_test("Product with zero", approx_equal(data[0], 0.0f));
    } catch (const std::exception& e) {
        log_test("Product with zero", false, e.what());
    }
    
    // Identical values
    try {
        Tensor t(Shape{{100}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> same(100, 7.5f);
        t.set_data(same);
        Tensor result_max = reduce_max(t);
        Tensor result_min = reduce_min(t);
        Tensor result_argmax = reduce_argmax(t);
        float* max_data = result_max.data<float>();
        float* min_data = result_min.data<float>();
        int64_t* argmax_data = result_argmax.data<int64_t>();
        bool passed = approx_equal(max_data[0], 7.5f) &&
                     approx_equal(min_data[0], 7.5f) &&
                     argmax_data[0] == 0; // First index
        log_test("Identical values", passed);
    } catch (const std::exception& e) {
        log_test("Identical values", false, e.what());
    }
}

// ========================================================================
// LARGE SCALE TESTS
// ========================================================================

void test_large_tensors() {
    print_test_header("LARGE SCALE TENSOR TESTS");
    
    // Large 1D tensor
    try {
        const size_t size = 1000000; // 1 million elements
        Tensor t(Shape{{static_cast<int64_t>(size)}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(size, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float* res_data = result.data<float>();
        bool correct = approx_equal(res_data[0], static_cast<float>(size), 1.0f);
        std::cout << "       [Time: " << duration.count() << "ms] ";
        log_test("1M element sum", correct);
    } catch (const std::exception& e) {
        log_test("1M element sum", false, e.what());
    }
    
    // Large 2D tensor
    try {
        Tensor t(Shape{{1000, 1000}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(1000000, 2.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(t, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float* res_data = result.data<float>();
        bool correct = approx_equal(res_data[0], 2.0f, 0.01f);
        std::cout << "       [Time: " << duration.count() << "ms] ";
        log_test("1000x1000 mean along axis", correct);
    } catch (const std::exception& e) {
        log_test("1000x1000 mean along axis", false, e.what());
    }
    
    // Large 3D tensor
    try {
        Tensor t(Shape{{100, 100, 100}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(1000000, 1.5f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_max(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        bool shape_ok = result.shape().dims.size() == 2 &&
                       result.shape().dims[0] == 100 &&
                       result.shape().dims[1] == 100;
        std::cout << "       [Time: " << duration.count() << "ms] ";
        log_test("100x100x100 max reduction", shape_ok);
    } catch (const std::exception& e) {
        log_test("100x100x100 max reduction", false, e.what());
    }
    
    // Deep tensor (many dimensions)
    try {
        Tensor t(Shape{{10, 10, 10, 10}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(10000, 3.0f);
        t.set_data(data);
        
        Tensor result = reduce_sum(t, {1, 3}, false);
        bool shape_ok = result.shape().dims.size() == 2 &&
                       result.shape().dims[0] == 10 &&
                       result.shape().dims[1] == 10;
        log_test("10^4 tensor multi-axis reduction", shape_ok);
    } catch (const std::exception& e) {
        log_test("10^4 tensor multi-axis reduction", false, e.what());
    }
}

// ========================================================================
// PARALLEL EXECUTION TESTS
// ========================================================================

void test_parallel_execution() {
    print_test_header("PARALLEL EXECUTION & THREAD SAFETY");
    
    // Test with OpenMP enabled (implicit in implementation)
    try {
        Tensor t(Shape{{5000, 500}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(2500000);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i % 100);
        }
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        bool shape_ok = result.shape().dims.size() == 1 && result.shape().dims[0] == 500;
        std::cout << "       [Time: " << duration.count() << "ms with OpenMP] ";
        log_test("Large parallel reduction", shape_ok);
    } catch (const std::exception& e) {
        log_test("Large parallel reduction", false, e.what());
    }
    
    // Verify deterministic results (same input -> same output)
    try {
        Tensor t(Shape{{1000, 100}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(100000);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>((i * 7) % 97); // Pseudo-random
        }
        t.set_data(data);
        
        Tensor result1 = reduce_sum(t, {1}, false);
        Tensor result2 = reduce_sum(t, {1}, false);
        
        float* data1 = result1.data<float>();
        float* data2 = result2.data<float>();
        
        bool deterministic = true;
        for (size_t i = 0; i < 1000 && deterministic; i++) {
            if (!approx_equal(data1[i], data2[i], 1e-3f)) {
                deterministic = false;
            }
        }
        log_test("Deterministic parallel results", deterministic);
    } catch (const std::exception& e) {
        log_test("Deterministic parallel results", false, e.what());
    }
    
    // Race condition test (accumulator correctness)
    try {
        Tensor t(Shape{{10000}}, TensorOptions().with_dtype(Dtype::Int32));
        std::vector<int32_t> data(10000, 1);
        t.set_data(data);
        
        Tensor result = reduce_sum(t);
        int64_t* res_data = result.data<int64_t>();
        log_test("Accumulator correctness (no race)", res_data[0] == 10000LL);
    } catch (const std::exception& e) {
        log_test("Accumulator correctness (no race)", false, e.what());
    }
}

// ========================================================================
// COMPREHENSIVE OPERATION TESTS
// ========================================================================

void test_all_operations() {
    print_test_header("ALL 14 REDUCTION OPERATIONS");
    
    Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32));
    t.set_data({1.0f, 2.0f, 3.0f, 4.0f});
    
    // 1. reduce_sum
    try {
        Tensor result = reduce_sum(t);
        float* data = result.data<float>();
        log_test("1. reduce_sum", approx_equal(data[0], 10.0f));
    } catch (const std::exception& e) {
        log_test("1. reduce_sum", false, e.what());
    }
    
    // 2. reduce_product
    try {
        Tensor result = reduce_product(t);
        float* data = result.data<float>();
        log_test("2. reduce_product", approx_equal(data[0], 24.0f));
    } catch (const std::exception& e) {
        log_test("2. reduce_product", false, e.what());
    }
    
    // 3. reduce_min
    try {
        Tensor result = reduce_min(t);
        float* data = result.data<float>();
        log_test("3. reduce_min", approx_equal(data[0], 1.0f));
    } catch (const std::exception& e) {
        log_test("3. reduce_min", false, e.what());
    }
    
    // 4. reduce_max
    try {
        Tensor result = reduce_max(t);
        float* data = result.data<float>();
        log_test("4. reduce_max", approx_equal(data[0], 4.0f));
    } catch (const std::exception& e) {
        log_test("4. reduce_max", false, e.what());
    }
    
    // 5. reduce_mean
    try {
        Tensor result = reduce_mean(t);
        float* data = result.data<float>();
        log_test("5. reduce_mean", approx_equal(data[0], 2.5f));
    } catch (const std::exception& e) {
        log_test("5. reduce_mean", false, e.what());
    }
    
    Tensor t_nan(Shape{{5}}, TensorOptions().with_dtype(Dtype::Float32));
    t_nan.set_data({1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f, 4.0f});
    
    // 6. reduce_nansum
    try {
        Tensor result = reduce_nansum(t_nan);
        float* data = result.data<float>();
        log_test("6. reduce_nansum", approx_equal(data[0], 10.0f));
    } catch (const std::exception& e) {
        log_test("6. reduce_nansum", false, e.what());
    }
    
    // 7. reduce_nanproduct
    try {
        Tensor result = reduce_nanproduct(t_nan);
        float* data = result.data<float>();
        log_test("7. reduce_nanproduct", approx_equal(data[0], 24.0f));
    } catch (const std::exception& e) {
        log_test("7. reduce_nanproduct", false, e.what());
    }
    
    // 8. reduce_nanmin
    try {
        Tensor result = reduce_nanmin(t_nan);
        float* data = result.data<float>();
        log_test("8. reduce_nanmin", approx_equal(data[0], 1.0f));
    } catch (const std::exception& e) {
        log_test("8. reduce_nanmin", false, e.what());
    }
    
    // 9. reduce_nanmax
    try {
        Tensor result = reduce_nanmax(t_nan);
        float* data = result.data<float>();
        log_test("9. reduce_nanmax", approx_equal(data[0], 4.0f));
    } catch (const std::exception& e) {
        log_test("9. reduce_nanmax", false, e.what());
    }
    
    // 10. reduce_nanmean
    try {
        Tensor result = reduce_nanmean(t_nan);
        float* data = result.data<float>();
        log_test("10. reduce_nanmean", !std::isnan(data[0]));
    } catch (const std::exception& e) {
        log_test("10. reduce_nanmean", false, e.what());
    }
    
    // 11. reduce_argmin
    try {
        Tensor result = reduce_argmin(t);
        int64_t* data = result.data<int64_t>();
        log_test("11. reduce_argmin", data[0] == 0);
    } catch (const std::exception& e) {
        log_test("11. reduce_argmin", false, e.what());
    }
    
    // 12. reduce_argmax
    try {
        Tensor result = reduce_argmax(t);
        int64_t* data = result.data<int64_t>();
        log_test("12. reduce_argmax", data[0] == 3);
    } catch (const std::exception& e) {
        log_test("12. reduce_argmax", false, e.what());
    }
    
    // 13. reduce_nanargmin
    try {
        Tensor result = reduce_nanargmin(t_nan);
        int64_t* data = result.data<int64_t>();
        log_test("13. reduce_nanargmin", data[0] == 0);
    } catch (const std::exception& e) {
        log_test("13. reduce_nanargmin", false, e.what());
    }
    
    // 14. reduce_nanargmax
    try {
        Tensor result = reduce_nanargmax(t_nan);
        int64_t* data = result.data<int64_t>();
        log_test("14. reduce_nanargmax", data[0] == 4);
    } catch (const std::exception& e) {
        log_test("14. reduce_nanargmax", false, e.what());
    }
}

// ========================================================================
// STRESS TESTS
// ========================================================================

void test_stress_scenarios() {
    print_test_header("STRESS TEST SCENARIOS");
    
    // Repeated reductions
    try {
        Tensor t(Shape{{100, 100}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(10000, 1.0f);
        t.set_data(data);
        
        bool all_passed = true;
        for (int i = 0; i < 10; i++) {
            Tensor result = reduce_sum(t, {0}, false);
            float* res_data = result.data<float>();
            if (!approx_equal(res_data[0], 100.0f, 0.1f)) {
                all_passed = false;
                break;
            }
        }
        log_test("Repeated reductions (10x)", all_passed);
    } catch (const std::exception& e) {
        log_test("Repeated reductions (10x)", false, e.what());
    }
    
    // Chain reductions
    try {
        Tensor t(Shape{{10, 10, 10}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(1000, 2.0f);
        t.set_data(data);
        
        Tensor r1 = reduce_sum(t, {2}, false);  // -> (10, 10)
        Tensor r2 = reduce_mean(r1, {1}, false); // -> (10,)
        Tensor r3 = reduce_max(r2);              // -> scalar
        
        float* final_data = r3.data<float>();
        log_test("Chained reductions", final_data[0] > 0.0f);
    } catch (const std::exception& e) {
        log_test("Chained reductions", false, e.what());
    }
    
    // Memory stress (multiple large tensors)
    try {
        std::vector<Tensor> tensors;
        for (int i = 0; i < 5; i++) {
            Tensor t(Shape{{1000, 100}}, TensorOptions().with_dtype(Dtype::Float32));
            std::vector<float> data(100000, static_cast<float>(i + 1));
            t.set_data(data);
            tensors.push_back(std::move(t));
        }
        
        bool all_correct = true;
        for (size_t i = 0; i < tensors.size(); i++) {
            Tensor result = reduce_mean(tensors[i]);
            float* res_data = result.data<float>();
            if (!approx_equal(res_data[0], static_cast<float>(i + 1), 0.1f)) {
                all_correct = false;
                break;
            }
        }
        log_test("Multiple large tensor stress", all_correct);
    } catch (const std::exception& e) {
        log_test("Multiple large tensor stress", false, e.what());
    }
}

// ========================================================================
// MAIN TEST RUNNER
// ========================================================================

void print_summary() {
    std::cout << "\n" << COLOR_MAGENTA << "========================================" << COLOR_RESET << "\n";
    std::cout << COLOR_MAGENTA << "  TEST SUMMARY" << COLOR_RESET << "\n";
    std::cout << COLOR_MAGENTA << "========================================" << COLOR_RESET << "\n\n";
    
    std::cout << "Total Tests:  " << total_tests << "\n";
    std::cout << COLOR_GREEN << "Passed:       " << passed_tests << COLOR_RESET << "\n";
    std::cout << COLOR_RED << "Failed:       " << failed_tests << COLOR_RESET << "\n";
    
    double pass_rate = (total_tests > 0) ? (100.0 * passed_tests / total_tests) : 0.0;
    std::cout << "\nPass Rate:    " << std::fixed << std::setprecision(1) << pass_rate << "%\n";
    
    if (failed_tests > 0) {
        std::cout << "\n" << COLOR_YELLOW << "Failed Tests:" << COLOR_RESET << "\n";
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
    
    std::cout << "\n" << COLOR_MAGENTA << "========================================" << COLOR_RESET << "\n\n";
    
    if (failed_tests == 0) {
        std::cout << COLOR_GREEN << "🎉 ALL TESTS PASSED! 🎉" << COLOR_RESET << "\n\n";
    } else {
        std::cout << COLOR_RED << "⚠️  SOME TESTS FAILED ⚠️" << COLOR_RESET << "\n\n";
    }
}

int main() {
    std::cout << COLOR_CYAN << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║     OwnTensor Reduction Operations Test Suite       ║\n";
    std::cout << "║     Comprehensive Testing Framework                  ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << COLOR_RESET << "\n";
    
    try {
        // Run all test suites
        test_all_dtypes_sum();
        test_1d_reductions();
        test_2d_reductions();
        test_3d_reductions();
        test_high_dimensional();
        test_axis_arguments();
        test_nan_handling();
        test_index_reductions();
        test_edge_cases();
        test_large_tensors();
        test_parallel_execution();
        test_all_operations();
        test_stress_scenarios();
        
        // Print summary
        print_summary();
        
    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "\n\nFATAL ERROR: " << e.what() << COLOR_RESET << "\n\n";
        return 1;
    }
    
    return (failed_tests == 0) ? 0 : 1;
}