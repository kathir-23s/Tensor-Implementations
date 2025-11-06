#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

#ifdef WITH_CUDA
  #include <cuda_runtime.h>
#endif

#include "TensorLib.h"
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
        std::ofstream file("local_test/" + report_filename);
        
        // Header
        file << "# Trigonometric & Hyperbolic Functions Test Report\n\n";
        file << "**Generated:** " << get_timestamp() << "\n\n";
        
        // Summary
        file << "## Summary\n\n";
        file << "| Metric | Value |\n";
        file << "|--------|-------|\n";
        file << "| Total Tests | " << total_tests << " |\n";
        file << "| Passed | " << passed_tests << " |\n";
        file << "| Failed | " << failed_tests << " |\n";
        file << "| Success Rate | " << std::fixed << std::setprecision(2)
             << (100.0 * (total_tests ? passed_tests / (double)total_tests : 0.0)) << "% |\n\n";
        
        // Detailed Results
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
        file << "- sin() / sin_()\n";
        file << "- cos() / cos_()\n";
        file << "- tan() / tan_()\n";
        file << "- asin() / asin_()\n";
        file << "- acos() / acos_()\n";
        file << "- atan() / atan_()\n";
        file << "- sinh() / sinh_()\n";
        file << "- cosh() / cosh_()\n";
        file << "- tanh() / tanh_()\n";
        file << "- asinh() / asinh_()\n";
        file << "- acosh() / acosh_()\n";
        file << "- atanh() / atanh_()\n\n";
        
        file << "### Devices Tested\n";
        file << "- CPU\n";
        file << "- GPU (CUDA)\n\n";
        
        file << "### Data Types Tested\n";
        file << "- Int16, Int32, Int64 (out-of-place only where meaningful)\n";
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
// Utility: build test name helper
// ============================================================================

static std::string build_test_name(const std::string& base, bool inplace, const Tensor& t, Dtype dt) {
    return base + (inplace ? "_" : "") + " (" + (t.is_cpu() ? "CPU" : "GPU") + ", " + get_dtype_name(dt) + ")";
}

// ============================================================================
// Generic unary test function
// ============================================================================

void test_unary_trig(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace,
                     const std::string& op_name,
                     const std::function<Tensor(const Tensor&)>& op_fn,
                     const std::function<void(Tensor&)>& op_inplace_fn,
                     const std::vector<float>& input_data,
                     const std::vector<float>& expected,
                     double atol = 1e-2) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = build_test_name(op_name, inplace, input, dtype);

        // For integer dtypes, in-place trig should be disallowed
        if (inplace && (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64)) {
            try {
                op_inplace_fn(input);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                report.add_result({test_name, false, "Expected exception for integer in-place operation", time_ms});
                return;
            } catch (const std::exception& e) {
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                report.add_result({test_name, true, "Correctly threw exception: " + std::string(e.what()), time_ms});
                return;
            }
        }

        if (inplace) {
            op_inplace_fn(input);
            bool passed = verify_tensor_values(input, expected, atol);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match expected" : "Values mismatch", time_ms});
        } else {
            Tensor output = op_fn(input);
            bool passed = verify_tensor_values(output, expected, atol);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match expected" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({op_name + std::string(inplace ? "_" : ""), false, std::string("Exception: ") + e.what(), time_ms});
    }
}

// ============================================================================
// Specific test wrappers
// ============================================================================

void test_sin(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 3.14159265f/6.0f, 3.14159265f/2.0f, 3.14159265f, -3.14159265f/4.0f};
    std::vector<float> exp = {0.0f, 0.5f, 1.0f, 0.0f, -0.70710678f};
    test_unary_trig(report, device, dtype, inplace, "sin",
                    [](const Tensor& t){ return sin(t); },
                    [](Tensor& t){ sin_(t); },
                    in, exp, 1e-3);
}

void test_cos(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 3.14159265f/3.0f, 3.14159265f/2.0f, 3.14159265f, -3.14159265f/4.0f};
    std::vector<float> exp = {1.0f, 0.5f, 0.0f, -1.0f, 0.70710678f};
    test_unary_trig(report, device, dtype, inplace, "cos",
                    [](const Tensor& t){ return cos(t); },
                    [](Tensor& t){ cos_(t); },
                    in, exp, 1e-3);
}

void test_tan(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 3.14159265f/6.0f, 3.14159265f/4.0f, -3.14159265f/4.0f, 0.3f};
    std::vector<float> exp = {0.0f, 0.57735027f, 1.0f, -1.0f, std::tan(0.3f)};
    test_unary_trig(report, device, dtype, inplace, "tan",
                    [](const Tensor& t){ return tan(t); },
                    [](Tensor& t){ tan_(t); },
                    in, exp, 1e-3);
}

void test_asin(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    std::vector<float> exp = {0.0f, std::asin(0.5f), std::asin(-0.5f), std::asin(1.0f), std::asin(-1.0f)};
    test_unary_trig(report, device, dtype, inplace, "asin",
                    [](const Tensor& t){ return asin(t); },
                    [](Tensor& t){ asin_(t); },
                    in, exp, 1e-3);
}

void test_acos(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {1.0f, 0.5f, 0.0f, -0.5f, -1.0f};
    std::vector<float> exp = {0.0f, std::acos(0.5f), std::acos(0.0f), std::acos(-0.5f), std::acos(-1.0f)};
    test_unary_trig(report, device, dtype, inplace, "acos",
                    [](const Tensor& t){ return acos(t); },
                    [](Tensor& t){ acos_(t); },
                    in, exp, 1e-3);
}

void test_atan(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 1.0f, -1.0f, 10.0f, -10.0f};
    std::vector<float> exp = {0.0f, std::atan(1.0f), std::atan(-1.0f), std::atan(10.0f), std::atan(-10.0f)};
    test_unary_trig(report, device, dtype, inplace, "atan",
                    [](const Tensor& t){ return atan(t); },
                    [](Tensor& t){ atan_(t); },
                    in, exp, 1e-3);
}

void test_sinh(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    std::vector<float> exp = {0.0f, std::sinh(0.5f), std::sinh(-0.5f), std::sinh(1.0f), std::sinh(-1.0f)};
    test_unary_trig(report, device, dtype, inplace, "sinh",
                    [](const Tensor& t){ return sinh(t); },
                    [](Tensor& t){ sinh_(t); },
                    in, exp, 1e-3);
}

void test_cosh(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 0.5f, -0.5f, 1.0f, 2.0f};
    std::vector<float> exp = {1.0f, std::cosh(0.5f), std::cosh(-0.5f), std::cosh(1.0f), std::cosh(2.0f)};
    test_unary_trig(report, device, dtype, inplace, "cosh",
                    [](const Tensor& t){ return cosh(t); },
                    [](Tensor& t){ cosh_(t); },
                    in, exp, 1e-3);
}

void test_tanh(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    std::vector<float> exp = {0.0f, std::tanh(0.5f), std::tanh(-0.5f), std::tanh(1.0f), std::tanh(-1.0f)};
    test_unary_trig(report, device, dtype, inplace, "tanh",
                    [](const Tensor& t){ return tanh(t); },
                    [](Tensor& t){ tanh_(t); },
                    in, exp, 1e-3);
}

void test_asinh(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    std::vector<float> in = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    std::vector<float> exp = {0.0f, std::asinh(0.5f), std::asinh(-0.5f), std::asinh(1.0f), std::asinh(-1.0f)};
    test_unary_trig(report, device, dtype, inplace, "asinh",
                    [](const Tensor& t){ return asinh(t); },
                    [](Tensor& t){ asinh_(t); },
                    in, exp, 1e-3);
}

void test_acosh(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    // acosh domain: x >= 1
    std::vector<float> in = {1.0f, 1.5f, 2.0f, 10.0f, 100.0f};
    std::vector<float> exp = {std::acosh(1.0f), std::acosh(1.5f), std::acosh(2.0f), std::acosh(10.0f), std::acosh(100.0f)};
    test_unary_trig(report, device, dtype, inplace, "acosh",
                    [](const Tensor& t){ return acosh(t); },
                    [](Tensor& t){ acosh_(t); },
                    in, exp, 1e-3);
}

void test_atanh(TestReport& report, const DeviceIndex& device, Dtype dtype, bool inplace) {
    // atanh domain: -1 < x < 1
    std::vector<float> in = {0.0f, 0.2f, -0.2f, 0.5f, -0.5f};
    std::vector<float> exp = {0.0f, std::atanh(0.2f), std::atanh(-0.2f), std::atanh(0.5f), std::atanh(-0.5f)};
    test_unary_trig(report, device, dtype, inplace, "atanh",
                    [](const Tensor& t){ return atanh(t); },
                    [](Tensor& t){ atanh_(t); },
                    in, exp, 1e-3);
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << R"(
========================================
  TRIGONOMETRIC & HYPERBOLIC FUNCTIONS - COMPREHENSIVE TEST SUITE
========================================

)";
    
    TestReport report("Trig_Hyperbolic_Report.md");
    
    std::vector<DeviceIndex> devices = {DeviceIndex(Device::CPU), DeviceIndex(Device::CUDA)};
    std::vector<Dtype> dtypes = {
        Dtype::Int16, Dtype::Int32, Dtype::Int64,
        Dtype::Float32, Dtype::Float64,
        Dtype::Float16, Dtype::Bfloat16
    };
    std::vector<bool> modes = {false, true};  // false = out-of-place, true = in-place
    
    int test_count = 0;
    int total_expected = (int)devices.size() * (int)dtypes.size() * (int)modes.size() * 12;  // 12 functions

    for (const auto& device : devices) {
        for (const auto& dtype : dtypes) {
            for (bool inplace : modes) {
                std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;

                test_sin(report, device, dtype, inplace);
                test_count++;

                test_cos(report, device, dtype, inplace);
                test_count++;

                test_tan(report, device, dtype, inplace);
                test_count++;

                test_asin(report, device, dtype, inplace);
                test_count++;

                test_acos(report, device, dtype, inplace);
                test_count++;

                test_atan(report, device, dtype, inplace);
                test_count++;

                test_sinh(report, device, dtype, inplace);
                test_count++;

                test_cosh(report, device, dtype, inplace);
                test_count++;

                test_tanh(report, device, dtype, inplace);
                test_count++;

                test_asinh(report, device, dtype, inplace);
                test_count++;

                test_acosh(report, device, dtype, inplace);
                test_count++;

                test_atanh(report, device, dtype, inplace);
                test_count++;
            }
        }
    }

    std::cout << "\rProgress: " << test_count << "/" << total_expected << " ✓\n\n";

    report.generate_markdown();

    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
