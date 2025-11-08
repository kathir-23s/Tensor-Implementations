#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <ctime>

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
        file << "# Arithmetic Functions Test Report\n\n";
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
        
        // Passed tests
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
        file << "- square() / square_()\n";
        file << "- square_root() / square_root_()\n";
        file << "- reciprocal() / reciprocal_()\n";
        file << "- negator() / negator_()\n";
        file << "- absolute() / absolute_()\n";
        file << "- sign() / sign_()\n\n";
        
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
// Test Functions for Each Operation
// ============================================================================

void test_square_function(TestReport& report, const DeviceIndex& device, 
                         Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = "square" + std::string(inplace ? "_" : "") + 
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            square_(input, 0);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, 
                             passed ? "Values match expected" : "Values mismatch", time_ms});
        } else {
            Tensor output = square(input, 0);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, 
                             passed ? "Values match expected" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"square" + std::string(inplace ? "_" : ""), false, 
                         std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_square_root_function(TestReport& report, const DeviceIndex& device,
                              Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> input_data = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f};
        std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = "square_root" + std::string(inplace ? "_" : "") + 
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            // Check for integer dtypes (should throw)
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    sqrt_(input,0);
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
            
            sqrt_(input, 0);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = sqrt(input, 0);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"square_root" + std::string(inplace ? "_" : ""), false, 
                         std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_reciprocal_function(TestReport& report, const DeviceIndex& device,
                             Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> input_data = {1.0f, 2.0f, 4.0f, 5.0f, 10.0f};
        std::vector<float> expected = {1.0f, 0.5f, 0.25f, 0.2f, 0.1f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = "reciprocal" + std::string(inplace ? "_" : "") + 
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            // Check for integer dtypes (should throw)
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    reciprocal_(input, 0);
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
            
            reciprocal_(input, 0);
            bool passed = verify_tensor_values(input, expected, 1e-2);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = reciprocal(input, 0);
            bool passed = verify_tensor_values(output, expected, 1e-2);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"reciprocal" + std::string(inplace ? "_" : ""), false, 
                         std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_negator_function(TestReport& report, const DeviceIndex& device,
                          Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        std::vector<float> expected = {2.0f, 1.0f, 0.0f, -1.0f, -2.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = "negator" + std::string(inplace ? "_" : "") + 
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            neg_(input, 0);
            bool passed = verify_tensor_values(input, expected, 1e-5);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = neg(input, 0);
            bool passed = verify_tensor_values(output, expected, 1e-5);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"negator" + std::string(inplace ? "_" : ""), false, 
                         std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_absolute_function(TestReport& report, const DeviceIndex& device,
                           Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> input_data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
        std::vector<float> expected = {5.0f, 2.0f, 0.0f, 2.0f, 5.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = "absolute" + std::string(inplace ? "_" : "") + 
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            abs_(input, 0);
            bool passed = verify_tensor_values(input, expected, 1e-5);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = abs(input, 0);
            bool passed = verify_tensor_values(output, expected, 1e-5);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"absolute" + std::string(inplace ? "_" : ""), false, 
                         std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_sign_function(TestReport& report, const DeviceIndex& device,
                       Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<float> input_data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
        std::vector<float> expected = {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f};
        
        Tensor input = create_tensor_from_float(input_data, device, dtype);
        std::string test_name = "sign" + std::string(inplace ? "_" : "") + 
                               " (" + (input.is_cpu() ? "CPU" : "GPU") + ", " + 
                               get_dtype_name(dtype) + ")";
        
        if (inplace) {
            sign_(input, 0);
            bool passed = verify_tensor_values(input, expected, 1e-5);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        } else {
            Tensor output = sign(input, 0);
            bool passed = verify_tensor_values(output, expected, 1e-5);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : "Values mismatch", time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"sign" + std::string(inplace ? "_" : ""), false, 
                         std::string("Exception: ") + e.what(), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << " ARITHMETIC FUNCTIONS - COMPREHENSIVE TEST SUITE\n";
    std::cout << "========================================\n\n";
    
    TestReport report("arith_test_report.md");
    
    // Define test configurations
    std::vector<DeviceIndex> devices = {DeviceIndex(Device::CPU), DeviceIndex(Device::CUDA)};
    std::vector<Dtype> dtypes = {
        Dtype::Int16, Dtype::Int32, Dtype::Int64,
        Dtype::Float32, Dtype::Float64,
        Dtype::Float16, Dtype::Bfloat16
    };
    std::vector<bool> modes = {false, true}; // false = out-of-place, true = in-place
    
    int test_count = 0;
    int total_expected = devices.size() * dtypes.size() * modes.size() * 6; // 6 functions
    
    // Run all tests
    for (const auto& device : devices) {
        for (const auto& dtype : dtypes) {
            for (bool inplace : modes) {
                std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;
                
                test_square_function(report, device, dtype, inplace);
                test_count++;
                
                test_square_root_function(report, device, dtype, inplace);
                test_count++;
                
                test_reciprocal_function(report, device, dtype, inplace);
                test_count++;
                
                test_negator_function(report, device, dtype, inplace);
                test_count++;
                
                test_absolute_function(report, device, dtype, inplace);
                test_count++;
                
                test_sign_function(report, device, dtype, inplace);
                test_count++;
            }
        }
    }
    
    std::cout << "\rProgress: " << test_count << "/" << total_expected << " ✓\n\n";
    
    // Generate report
    report.generate_markdown();
    
    std::cout << "\n========================================\n";
    std::cout << " ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
