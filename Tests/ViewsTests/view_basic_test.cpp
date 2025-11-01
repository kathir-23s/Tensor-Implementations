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
        file << "# View Operations Basic Test Report\n\n";
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
        file << "### Operations Tested\n";
        file << "- view()\n";
        file << "- reshape()\n";
        file << "- transpose()\n";
        file << "- t()\n";
        file << "- flatten()\n";
        file << "- unflatten()\n\n";
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
// Helper Functions for View Testing
// ============================================================================

bool verify_shape(const Tensor& t, const std::vector<int64_t>& expected_shape) {
    const auto& shape = t.shape().dims;
    return shape == expected_shape; // No type mismatch now!
}

template<typename T>
bool tensors_share_data(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype()) return false;
    return a.data<T>() == b.data<T>();
}

bool check_data_sharing(const Tensor& a, const Tensor& b, Dtype dtype) {
    switch(dtype) {
        case Dtype::Int16: return tensors_share_data<int16_t>(a, b);
        case Dtype::Int32: return tensors_share_data<int32_t>(a, b);
        case Dtype::Int64: return tensors_share_data<int64_t>(a, b);
        case Dtype::Float32: return tensors_share_data<float>(a, b);
        case Dtype::Float64: return tensors_share_data<double>(a, b);
        case Dtype::Float16: return tensors_share_data<float16_t>(a, b);
        case Dtype::Bfloat16: return tensors_share_data<bfloat16_t>(a, b);
        default: return false;
    }
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

void test_view(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor original = create_tensor_from_float(data, device, dtype);
        original = original.reshape({{2, 6}});
        Tensor viewed = original.view({{3, 4}});
        std::string test_name = std::string("view (")+ (original.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dtype) + ")";
        bool shape_correct = verify_shape(viewed, {3, 4});
        bool shares_data = check_data_sharing(original, viewed, dtype);
        bool passed = shape_correct && shares_data;
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string message = passed ? "Shape correct, data sharing verified" : "Shape=" + std::to_string(shape_correct) + ", Sharing=" + std::to_string(shares_data);
        report.add_result({test_name, passed, message, time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = "view (" + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_reshape(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor original = create_tensor_from_float(data, device, dtype);
        original = original.reshape({{2, 6}});
        Tensor reshaped = original.reshape({{3, 4}});
        std::string test_name = std::string("reshape (") + (original.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dtype) + ")";
        bool shape_correct = verify_shape(reshaped, {3, 4});
        bool shares_data = check_data_sharing(original, reshaped, dtype);
        bool passed = shape_correct && shares_data;
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string message = passed ? "Reshape verified" : "Shape=" + std::to_string(shape_correct) + ", Sharing=" + std::to_string(shares_data);
        report.add_result({test_name, passed, message, time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = "reshape (" + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_transpose(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6};
        Tensor original = create_tensor_from_float(data, device, dtype);
        original = original.reshape({{2, 3}});
        Tensor transposed = original.transpose(0, 1);
        std::string test_name = std::string("transpose (") + (original.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dtype) + ")";
        bool shape_correct = verify_shape(transposed, {3, 2});
        bool shares_data = check_data_sharing(original, transposed, dtype);
        bool passed = shape_correct && shares_data;
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string message = passed ? "Transpose verified" : "Shape=" + std::to_string(shape_correct) + ", Sharing=" + std::to_string(shares_data);
        report.add_result({test_name, passed, message, time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = "transpose (" + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_t(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6};
        Tensor original = create_tensor_from_float(data, device, dtype);
        original = original.reshape({{2, 3}});
        Tensor transposed = original.t();
        std::string test_name = std::string("t() (") + (original.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dtype) + ")";
        bool shape_correct = verify_shape(transposed, {3, 2});
        bool shares_data = check_data_sharing(original, transposed, dtype);
        bool passed = shape_correct && shares_data;
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string message = passed ? "t() verified" : "Shape=" + std::to_string(shape_correct) + ", Sharing=" + std::to_string(shares_data);
        report.add_result({test_name, passed, message, time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = "t() (" + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_flatten(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor original = create_tensor_from_float(data, device, dtype);
        original = original.reshape({{2, 2, 3}});
        Tensor flattened = original.flatten();
        std::string test_name = std::string("flatten (") + (original.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dtype) + ")";
        bool shape_correct = verify_shape(flattened, {12});
        bool shares_data = check_data_sharing(original, flattened, dtype);
        bool passed = shape_correct && shares_data;
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string message = passed ? "Flatten verified" : "Shape=" + std::to_string(shape_correct) + ", Sharing=" + std::to_string(shares_data);
        report.add_result({test_name, passed, message, time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = "flatten (" + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_unflatten(TestReport& report, const DeviceIndex& device, Dtype dtype) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> data = {1,2,3,4,5,6,7,8,9,10,11,12};
        Tensor original = create_tensor_from_float(data, device, dtype);
        Tensor unflattened = original.unflatten(0, {{3, 4}});
        std::string test_name = std::string("unflatten (") + (original.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dtype) + ")";
        bool shape_correct = verify_shape(unflattened, {3, 4});
        bool shares_data = check_data_sharing(original, unflattened, dtype);
        bool passed = shape_correct && shares_data;
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string message = passed ? "Unflatten verified" : "Shape=" + std::to_string(shape_correct) + ", Sharing=" + std::to_string(shares_data);
        report.add_result({test_name, passed, message, time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string test_name = "unflatten (" + get_dtype_name(dtype) + ")";
        report.add_result({test_name, false, std::string("Exception: ") + e.what(), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  VIEW OPERATIONS - BASIC TEST SUITE\n";
    std::cout << "========================================\n\n";
    
    TestReport report("view_basic_testreport.md");
    
    std::vector<DeviceIndex> devices = {DeviceIndex(Device::CPU), DeviceIndex(Device::CUDA)};
    std::vector<Dtype> dtypes = {
        Dtype::Int16, Dtype::Int32, Dtype::Int64,
        Dtype::Float32, Dtype::Float64,
        Dtype::Float16, Dtype::Bfloat16
    };
    
    int test_count = 0;
    int total_expected = devices.size() * dtypes.size() * 6;  // 6 operations
    
    for (const auto& device : devices) {
        for (const auto& dtype : dtypes) {
            std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;
            test_view(report, device, dtype); test_count++;
            test_reshape(report, device, dtype); test_count++;
            test_transpose(report, device, dtype); test_count++;
            test_t(report, device, dtype); test_count++;
            test_flatten(report, device, dtype); test_count++;
            test_unflatten(report, device, dtype); test_count++;
        }
    }
    
    std::cout << "\rProgress: " << test_count << "/" << total_expected << " ✓\n\n";
    report.generate_markdown();
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    return 0;
}
