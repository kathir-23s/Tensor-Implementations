#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "Tensor.h"
#include "Types.h"
#include "ops/helpers/testutils.h"

using namespace OwnTensor;
using namespace TestUtils;

// ============================================================================
// Test Infrastructure
// ============================================================================

struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double execution_time_ms;
};

class TestReport {
private:
    std::vector<TestResult> results;
    std::string report_filename;
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

public:
    TestReport(const std::string& filename) : report_filename(filename) {}
    
    void add_result(const TestResult& result) {
        results.push_back(result);
        total_tests++;
        if (result.passed) {
            passed_tests++;
        } else {
            failed_tests++;
        }
    }
    
    void generate_markdown() {
        std::ofstream file(report_filename);
        file << "# View Operations Chained Test Report\n\n";
        file << "**Generated:** " << get_timestamp() << "\n\n";
        file << "## Summary\n\n";
        file << "| Metric | Value |\n";
        file << "|--------|-------|\n";
        file << "| Total Tests | " << total_tests << " |\n";
        file << "| Passed | " << passed_tests << " |\n";
        file << "| Failed | " << failed_tests << " |\n";
        file << "| Success Rate | " << std::fixed << std::setprecision(2)
             << (100.0 * passed_tests / total_tests) << "% |\n\n";
        file << "## Detailed Test Results\n\n";
        file << "### ✅ Passed Tests (" << passed_tests << ")\n\n";
        for (const auto& result : results) {
            if (result.passed) {
                file << "- **" << result.test_name << "** ("
                     << std::fixed << std::setprecision(3)
                     << result.execution_time_ms << " ms)\n";
                if (!result.message.empty()) {
                    file << "  - " << result.message << "\n";
                }
            }
        }
        file << "\n### ❌ Failed Tests (" << failed_tests << ")\n\n";
        if (failed_tests == 0) {
            file << "*No failed tests!*\n\n";
        } else {
            for (const auto& result : results) {
                if (!result.passed) {
                    file << "- **" << result.test_name << "**\n";
                    file << "  - Error: " << result.message << "\n";
                    file << "  - Execution time: " << std::fixed << std::setprecision(3)
                         << result.execution_time_ms << " ms\n";
                }
            }
        }
        file << "\n## Test Coverage\n\n";
        file << "### Chained Operations Tested\n";
        file << "- reshape → transpose → flatten\n";
        file << "- flatten → unflatten → t\n";
        file << "- view → reshape → flatten → unflatten\n";
        file << "- Complex multi-step chains\n\n";
        file << "### Devices Tested\n";
        file << "- CPU\n";
        file << "- GPU (CUDA)\n\n";
        file << "### Data Types Tested\n";
        file << "- Int16, Int32, Int64\n";
        file << "- Float32, Float64\n";
        file << "- Float16, Bfloat16\n\n";
        file.close();
        std::cout << "\n✅ Test report generated: " << report_filename << "\n";
    }
    
private:
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

bool verify_shape(const Tensor& t, const std::vector<int64_t>& expected_shape) {
    const auto& shape = t.shape().dims;
    return shape == expected_shape;
}

// ============================================================================
// Chained Operations Tests
// ============================================================================

void test_reshape_transpose_flatten(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor t = create_tensor_from_float(data, device, dtype);
        
        // Chain: [12] → reshape [2,6] → transpose [6,2] → flatten [12]
        t = t.reshape({{2, 6}}).transpose(0, 1).flatten();
        
        std::string test_name = std::string("reshape→transpose→flatten (") + 
                               (t.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        bool shape_correct = verify_shape(t, {12});
        bool passed = shape_correct;
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::string message = passed ? "Final shape [12] verified" : "Shape mismatch";
        report.add_result({test_name, passed, message, time_ms});
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = std::string("reshape→transpose→flatten (") + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_flatten_unflatten_t(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor t = create_tensor_from_float(data, device, dtype);
        
        // Chain: [12] → reshape [3,4] → t [4,3] → flatten [12] → unflatten [2,6]
        t = t.reshape({{3, 4}}).t().flatten().unflatten(0, {{2, 6}});
        
        std::string test_name = std::string("reshape→t→flatten→unflatten (") + 
                               (t.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        bool shape_correct = verify_shape(t, {2, 6});
        bool passed = shape_correct;
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::string message = passed ? "Final shape [2,6] verified" : "Shape mismatch";
        report.add_result({test_name, passed, message, time_ms});
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = std::string("reshape→t→flatten→unflatten (") + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_view_reshape_flatten_unflatten(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
        Tensor t = create_tensor_from_float(data, device, dtype);
        
        // Chain: [24] → reshape [4,6] → view [2,12] → flatten [24] → unflatten [3,8]
        t = t.reshape({{4, 6}}).view({{2, 12}}).flatten().unflatten(0, {{3, 8}});
        
        std::string test_name = std::string("view→reshape→flatten→unflatten (") + 
                               (t.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        bool shape_correct = verify_shape(t, {3, 8});
        bool passed = shape_correct;
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::string message = passed ? "Final shape [3,8] verified" : "Shape mismatch";
        report.add_result({test_name, passed, message, time_ms});
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = std::string("view→reshape→flatten→unflatten (") + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_complex_chain(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
        Tensor t = create_tensor_from_float(data, device, dtype);
        
        // Complex chain: [24] → reshape [2,3,4] → flatten [24] → reshape [4,6] → t [6,4] → flatten [24] → unflatten [2,12]
        t = t.reshape({{2, 3, 4}}).flatten().reshape({{4, 6}}).t().flatten().unflatten(0, {{2, 12}});
        
        std::string test_name = std::string("complex_chain (") + 
                               (t.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        bool shape_correct = verify_shape(t, {2, 12});
        bool passed = shape_correct;
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::string message = passed ? "Complex chain completed, shape [2,12] verified" : "Shape mismatch";
        report.add_result({test_name, passed, message, time_ms});
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = std::string("complex_chain (") + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_double_transpose(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6};
        Tensor t = create_tensor_from_float(data, device, dtype);
        
        // Chain: [6] → reshape [2,3] → t [3,2] → t [2,3]
        t = t.reshape({{2, 3}}).t().t();
        
        std::string test_name = std::string("double_transpose (") + 
                               (t.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        // Double transpose should return to original shape [2,3]
        bool shape_correct = verify_shape(t, {2, 3});
        bool passed = shape_correct;
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::string message = passed ? "Double transpose returns to [2,3]" : "Shape mismatch";
        report.add_result({test_name, passed, message, time_ms});
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = std::string("double_transpose (") + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_flatten_unflatten_roundtrip(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor t = create_tensor_from_float(data, device, dtype);
        
        // Chain: [12] → reshape [3,4] → flatten [12] → unflatten [3,4]
        t = t.reshape({{3, 4}}).flatten().unflatten(0, {{3, 4}});
        
        std::string test_name = std::string("flatten→unflatten_roundtrip (") + 
                               (t.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        // Should return to [3,4]
        bool shape_correct = verify_shape(t, {3, 4});
        bool passed = shape_correct;
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::string message = passed ? "Roundtrip returns to [3,4]" : "Shape mismatch";
        report.add_result({test_name, passed, message, time_ms});
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = std::string("flatten→unflatten_roundtrip (") + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  VIEW OPERATIONS - CHAINED TEST SUITE\n";
    std::cout << "========================================\n\n";
    
    TestReport report("view_chained_testreport.md");
    
    std::vector<DeviceIndex> devices = {DeviceIndex(Device::CPU), DeviceIndex(Device::CUDA)};
    std::vector<Dtype> dtypes = {
        Dtype::Int16, Dtype::Int32, Dtype::Int64,
        Dtype::Float32, Dtype::Float64,
        Dtype::Float16, Dtype::Bfloat16
    };
    
    int test_count = 0;
    int total_expected = devices.size() * dtypes.size() * 6;  // 6 chained operation tests
    
    for (const auto& device : devices) {
        for (const auto& dtype : dtypes) {
            std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;
            
            test_reshape_transpose_flatten(report, device, dtype); test_count++;
            test_flatten_unflatten_t(report, device, dtype); test_count++;
            test_view_reshape_flatten_unflatten(report, device, dtype); test_count++;
            test_complex_chain(report, device, dtype); test_count++;
            test_double_transpose(report, device, dtype); test_count++;
            test_flatten_unflatten_roundtrip(report, device, dtype); test_count++;
        }
    }
    
    std::cout << "\rProgress: " << test_count << "/" << total_expected << " ✓\n\n";
    report.generate_markdown();
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    return 0;
}