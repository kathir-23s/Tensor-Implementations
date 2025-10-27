#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

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
        file << "# Exponentiation & Logarithmic Functions Test Report\n\n";
        file << "**Generated:** " << get_timestamp() << "\n\n";
        
        // Summary
        file << "## Summary\n\n";
        file << "| Metric | Value |\n";
        file << "|--------|-------|\n";
        file << "| Total Tests | " << total_tests << " |\n";
        file << "| Passed | " << passed_tests << " |\n";
        file << "| Failed | " << failed_tests << " |\n";
        file << "| Success Rate | " << std::fixed << std::setprecision(2)
             << (100.0 * passed_tests / total_tests) << "% |\n\n";
        
        // Detailed Results
        file << "## Detailed Test Results\n\n";
        
        // Group by status
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
        
        // Test Coverage
        file << "\n## Test Coverage\n\n";
        file << "### Operations Tested\n";
        file << "- exp() / exp_()\n";
        file << "- exp2() / exp2_()\n";
        file << "- log() / log_()\n";
        file << "- log2() / log2_()\n";
        file << "- log10() / log10_()\n\n";
        
        file << "### Devices Tested\n";
        file << "- CPU\n";
        file << "- GPU (CUDA)\n\n";
        
        file << "### Data Types Tested\n";
        file << "- Int16, Int32, Int64 (out-of-place only)\n";
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
// Test Functions
// ============================================================================

void test_exp_function(TestReport& report, const DeviceIndex& device,
                      Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 1.0f};
        std::vector<float> expected = {1.0f, 2.71828f, 7.38906f, 20.0855f, 2.71828f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        
        // NOW construct test_name using tensor's methods
        std::string test_name = "exp" + std::string(inplace ? "_" : "") +
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " +
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            // Test in-place operation
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                // Should throw error
                try {
                    exp_(input);
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
            
            exp_(input);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            report.add_result({test_name, passed,
                             passed ? "Values match expected" : "Values mismatch", time_ms});
        } else {
            // Test out-of-place operation
            Tensor output = exp(input);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            report.add_result({test_name, passed,
                             passed ? "Values match expected" : "Values mismatch", time_ms});
        }
        
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"exp" + std::string(inplace ? "_" : ""), false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_exp2_function(TestReport& report, const DeviceIndex& device,
                       Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> expected = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        
        std::string test_name = "exp2" + std::string(inplace ? "_" : "") +
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " +
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    exp2_(input);
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, false, "Expected exception for integer in-place", time_ms});
                    return;
                } catch (const std::exception& e) {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, true, "Correctly threw exception", time_ms});
                    return;
                }
            }
            
            exp2_(input);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = exp2(input);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"exp2" + std::string(inplace ? "_" : ""), false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_log_function(TestReport& report, const DeviceIndex& device,
                      Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::vector<float> input_data = {1.0f, 3.0f, 7.0f, 20.0f, 148.0f};  // Rounded values
        std::vector<float> expected = {0.0f, 1.0986f, 1.9459f, 2.9957f, 4.9972f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        
        std::string test_name = "log" + std::string(inplace ? "_" : "") +
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " +
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    log_(input);
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, false, "Expected exception for integer in-place", time_ms});
                    return;
                } catch (const std::exception& e) {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, true, "Correctly threw exception", time_ms});
                    return;
                }
            }
            
            log_(input);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = log(input);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"log" + std::string(inplace ? "_" : ""), false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_log2_function(TestReport& report, const DeviceIndex& device,
                       Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::vector<float> input_data = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
        std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        
        std::string test_name = "log2" + std::string(inplace ? "_" : "") +
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " +
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    log2_(input);
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, false, "Expected exception for integer in-place", time_ms});
                    return;
                } catch (const std::exception& e) {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, true, "Correctly threw exception", time_ms});
                    return;
                }
            }
            
            log2_(input);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = log2(input);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"log2" + std::string(inplace ? "_" : ""), false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_log10_function(TestReport& report, const DeviceIndex& device,
                        Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::vector<float> input_data = {1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f};
        std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        
        std::string test_name = "log10" + std::string(inplace ? "_" : "") +
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " +
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    log10_(input);
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, false, "Expected exception for integer in-place", time_ms});
                    return;
                } catch (const std::exception& e) {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    report.add_result({test_name, true, "Correctly threw exception", time_ms});
                    return;
                }
            }
            
            log10_(input);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = log10(input);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"log10" + std::string(inplace ? "_" : ""), false, std::string("Exception: ") + e.what(), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  EXP/LOG FUNCTIONS - COMPREHENSIVE TEST SUITE\n";
    std::cout << "========================================\n\n";
    
    TestReport report("ExpLog_Basic_Report.md");
    
    // Define test configurations
    std::vector<DeviceIndex> devices = {DeviceIndex(Device::CPU), DeviceIndex(Device::CUDA)};
    std::vector<Dtype> dtypes = {
        Dtype::Int16, Dtype::Int32, Dtype::Int64,
        Dtype::Float32, Dtype::Float64,
        Dtype::Float16, Dtype::Bfloat16
    };
    std::vector<bool> modes = {false, true};  // false = out-of-place, true = in-place
    
    int test_count = 0;
    int total_expected = devices.size() * dtypes.size() * modes.size() * 5;  // 5 functions
    
    // Run all tests
    for (const auto& device : devices) {
        for (const auto& dtype : dtypes) {
            for (bool inplace : modes) {
                std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;
                
                test_exp_function(report, device, dtype, inplace);
                test_count++;
                
                test_exp2_function(report, device, dtype, inplace);
                test_count++;
                
                test_log_function(report, device, dtype, inplace);
                test_count++;
                
                test_log2_function(report, device, dtype, inplace);
                test_count++;
                
                test_log10_function(report, device, dtype, inplace);
                test_count++;
            }
        }
    }
    
    std::cout << "\rProgress: " << test_count << "/" << total_expected << " ✓\n\n";
    
    // Generate report
    report.generate_markdown();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}