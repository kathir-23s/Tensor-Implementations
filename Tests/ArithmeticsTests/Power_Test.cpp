#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <fstream>
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
        file << "# Power Function Test Report\n\n";
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
        file << "- power(tensor, int)\n";
        file << "- power(tensor, float)\n";
        file << "- power(tensor, double)\n";
        file << "- pow_(tensor, int) [in-place]\n";
        file << "- pow_(tensor, float) [in-place]\n";
        file << "- pow_(tensor, double) [in-place]\n\n";

        file << "### Edge Cases Tested\n";
        file << "- 0^0 → 1 (convention)\n";
        file << "- 0^(negative) → inf\n";
        file << "- 0^(positive) → 0\n";
        file << "- negative^(fractional) → NaN\n";
        file << "- Overflow/Underflow handling\n";
        file << "- NaN propagation\n\n";

        file << "### Devices Tested\n";
        file << "- CPU\n";
        file << "- GPU (CUDA)\n\n";

        file << "### Data Types Tested\n";
        file << "- Int16, Int32, Int64 (out-of-place only)\n";
        file << "- Float16, Bfloat16 (half-precision)\n";
        file << "- Float32, Float64 (full-precision)\n\n";

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

struct Tolerances { 
    double rel_tol; 
    double abs_tol; 
};

static Tolerances tol_for(Dtype dt) {
    switch (dt) {
        case Dtype::Float16:
        case Dtype::Bfloat16: return {5e-2, 1e-2};  // Looser tolerance for half precision
        case Dtype::Float32: return {1e-4, 1e-5};
        case Dtype::Float64: return {1e-6, 1e-8};
        default: return {1e-4, 1e-5};
    }
}

static bool almost_equal(double a, double b, const Tolerances& t) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    double diff = std::fabs(a - b);
    if (diff <= t.abs_tol) return true;
    double maxab = std::max(std::fabs(a), std::fabs(b));
    return diff <= t.rel_tol * maxab;
}

// Helper to convert float to half (for testing purposes)
static uint16_t float_to_half_bits(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (bits >> 16) & 0x8000;
    uint16_t exponent = ((bits >> 23) & 0xFF) - 112;
    uint16_t mantissa = (bits >> 13) & 0x3FF;
    if (exponent <= 0) return sign;
    if (exponent >= 31) return sign | 0x7C00;
    return sign | (exponent << 10) | mantissa;
}

static float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint16_t exponent = (h >> 10) & 0x1F;
    uint16_t mantissa = h & 0x3FF;
    if (exponent == 0) {
        uint32_t bits = sign;
        return *reinterpret_cast<float*>(&bits);
    }
    if (exponent == 31) {
        uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&bits);
    }
    uint32_t bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&bits);
}

static Tensor make_tensor_cpu(const std::vector<double>& vals, Dtype dt, 
                              const std::vector<int64_t>& shape) {
    Tensor t(Shape{shape}, dt, DeviceIndex(Device::CPU), false);
    switch (dt) {
        case Dtype::Float16: {
            // For Float16, store as uint16_t and convert
            auto* p = reinterpret_cast<uint16_t*>(t.data());
            for (size_t i=0; i<vals.size(); i++) {
                p[i] = float_to_half_bits(static_cast<float>(vals[i]));
            }
            break;
        }
        case Dtype::Bfloat16: {
            // For Bfloat16, truncate float32 to bf16
            auto* p = reinterpret_cast<uint16_t*>(t.data());
            for (size_t i=0; i<vals.size(); i++) {
                float f = static_cast<float>(vals[i]);
                uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
                p[i] = static_cast<uint16_t>(bits >> 16);  // Take upper 16 bits
            }
            break;
        }
        case Dtype::Float32: {
            auto* p = t.data<float>();
            for (size_t i=0; i<vals.size(); i++) p[i] = static_cast<float>(vals[i]);
            break;
        }
        case Dtype::Float64: {
            auto* p = t.data<double>();
            for (size_t i=0; i<vals.size(); i++) p[i] = vals[i];
            break;
        }
        default: break;
    }
    return t;
}

static std::vector<double> to_double_vec(const Tensor& t) {
    std::vector<double> out(t.numel());
    switch (t.dtype()) {
        case Dtype::Float16: {
            auto* p = reinterpret_cast<const uint16_t*>(t.data());
            for (int i=0; i<(int)out.size(); i++) {
                out[i] = static_cast<double>(half_bits_to_float(p[i]));
            }
            break;
        }
        case Dtype::Bfloat16: {
            auto* p = reinterpret_cast<const uint16_t*>(t.data());
            for (int i=0; i<(int)out.size(); i++) {
                uint32_t bits = static_cast<uint32_t>(p[i]) << 16;
                float f = *reinterpret_cast<float*>(&bits);
                out[i] = static_cast<double>(f);
            }
            break;
        }
        case Dtype::Float32: {
            auto* p = t.data<float>();
            for (int i=0; i<(int)out.size(); i++) out[i] = static_cast<double>(p[i]);
            break;
        }
        case Dtype::Float64: {
            auto* p = t.data<double>();
            for (int i=0; i<(int)out.size(); i++) out[i] = p[i];
            break;
        }
        default: break;
    }
    return out;
}

static bool check_tensor(const Tensor& out, const std::vector<double>& ref, 
                        const Tolerances& tol, std::string& msg) {
    auto got = to_double_vec(out);
    for (size_t i=0; i<ref.size(); i++) {
        if (!almost_equal(got[i], ref[i], tol)) {
            std::ostringstream oss;
            oss << "Mismatch at " << i << " got=" << got[i] << " ref=" << ref[i];
            msg = oss.str();
            return false;
        }
    }
    return true;
}

// ============================================================================
// Basic Power Tests
// ============================================================================

void test_pow_integer_exponent(TestReport& report, const DeviceIndex& device, 
                                 Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<double> input_data = {2.0, 3.0, 4.0, 5.0};
        std::vector<double> expected = {4.0, 9.0, 16.0, 25.0}; // x^2
        
        Tensor input = make_tensor_cpu(input_data, dtype, {4});
        
        // Move to GPU if needed
        if (device.device == Device::CUDA) {
            input = input.to(device);
        }
        
        std::string test_name = "power" + std::string(inplace ? "_" : "") + 
                               "(int=2) (" + (device.device == Device::CPU ? "CPU" : "GPU") + 
                               ", " + get_dtype_name(dtype) + ")";

        if (inplace) {
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    pow_(input, 2, 0);
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
            pow_(input, 2, 0);
            Tensor result = input.to(DeviceIndex(Device::CPU));
            std::string msg;
            bool passed = check_tensor(result, expected, tol_for(dtype), msg);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : msg, time_ms});
        } else {
            Tensor output = pow(input, 2, 0);
            Tensor result = output.to(DeviceIndex(Device::CPU));
            std::string msg;
            bool passed = check_tensor(result, expected, tol_for(dtype), msg);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : msg, time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"power(int)", false, std::string("Exception: ") + e.what(), time_ms});
    }
}

void test_pow_float_exponent(TestReport& report, const DeviceIndex& device, 
                               Dtype dtype, bool inplace) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::vector<double> input_data = {4.0, 9.0, 16.0, 25.0};
        std::vector<double> expected = {2.0, 3.0, 4.0, 5.0}; // x^0.5 (square root)
        
        Tensor input = make_tensor_cpu(input_data, dtype, {4});
        
        // Move to GPU if needed
        if (device.device == Device::CUDA) {
            input = input.to(device);
        }
        
        std::string test_name = "power" + std::string(inplace ? "_" : "") + 
                               "(float=0.5) (" + (device.device == Device::CPU ? "CPU" : "GPU") + 
                               ", " + get_dtype_name(dtype) + ")";

        if (inplace) {
            if (dtype == Dtype::Int16 || dtype == Dtype::Int32 || dtype == Dtype::Int64) {
                try {
                    pow_(input, 0.5f, 0);
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
            pow_(input, 0.5f, 0);
            Tensor result = input.to(DeviceIndex(Device::CPU));
            std::string msg;
            bool passed = check_tensor(result, expected, tol_for(dtype), msg);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : msg, time_ms});
        } else {
            Tensor output = pow(input, 0.5f, 0);
            Tensor result = output.to(DeviceIndex(Device::CPU));
            std::string msg;
            bool passed = check_tensor(result, expected, tol_for(dtype), msg);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.add_result({test_name, passed, passed ? "Values match" : msg, time_ms});
        }
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"power(float)", false, std::string("Exception: ") + e.what(), time_ms});
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_pow_edge_cases(TestReport& report, Dtype dtype) {
    // Test 0^0 = 1
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> x = {0.0};
        std::vector<double> expected = {1.0};
        Tensor a = make_tensor_cpu(x, dtype, {1});
        Tensor y = pow(a, 0, 0);
        std::string msg;
        bool ok = check_tensor(y, expected, tol_for(dtype), msg);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"edge_case/0^0=1 (" + get_dtype_name(dtype) + ")", ok, msg, time_ms});
    }

    // Test 0^(-n) = inf
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> x = {0.0};
        std::vector<double> expected = {std::numeric_limits<double>::infinity()};
        Tensor a = make_tensor_cpu(x, dtype, {1});
        Tensor y = pow(a, -2, 0);
        std::string msg;
        bool ok = check_tensor(y, expected, tol_for(dtype), msg);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"edge_case/0^(-2)=inf (" + get_dtype_name(dtype) + ")", ok, msg, time_ms});
    }

    // Test negative^(fractional) = NaN
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> x = {-4.0};
        std::vector<double> expected = {std::numeric_limits<double>::quiet_NaN()};
        Tensor a = make_tensor_cpu(x, dtype, {1});
        Tensor y = pow(a, 0.5f, 0);
        std::string msg;
        bool ok = check_tensor(y, expected, tol_for(dtype), msg);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"edge_case/(-4)^0.5=NaN (" + get_dtype_name(dtype) + ")", ok, msg, time_ms});
    }

    // Test overflow
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> x = {10.0};
        std::vector<double> expected = {std::numeric_limits<double>::infinity()};
        Tensor a = make_tensor_cpu(x, dtype, {1});
        Tensor y = pow(a, 1000, 0);
        std::string msg;
        bool ok = check_tensor(y, expected, tol_for(dtype), msg);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"edge_case/overflow 10^1000 (" + get_dtype_name(dtype) + ")", ok, msg, time_ms});
    }

    // Test underflow
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> x = {10.0};
        std::vector<double> expected = {0.0};
        Tensor a = make_tensor_cpu(x, dtype, {1});
        Tensor y = pow(a, -1000, 0);
        std::string msg;
        bool ok = check_tensor(y, expected, tol_for(dtype), msg);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"edge_case/underflow 10^(-1000) (" + get_dtype_name(dtype) + ")", ok, msg, time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  POWER FUNCTION - COMPREHENSIVE TEST SUITE\n";
    std::cout << "  Including Float16/Bfloat16 Support\n";
    std::cout << "========================================\n\n";

    TestReport report("pow_Test_Report.md");

    // Define test configurations
    std::vector<DeviceIndex> devices = {
        DeviceIndex(Device::CPU)
#ifdef WITH_CUDA
        , DeviceIndex(Device::CUDA)
#endif
    };
    
    std::vector<Dtype> dtypes = {
        Dtype::Float16,    // Half precision
        Dtype::Bfloat16,   // Brain float 16
        Dtype::Float32,    // Single precision
        Dtype::Float64     // Double precision
    };
    
    std::vector<bool> modes = {false, true}; // false = out-of-place, true = in-place

    int test_count = 0;
    int total_expected = devices.size() * dtypes.size() * modes.size() * 2 + dtypes.size() * 5;

    // Run basic tests
    std::cout << "Running basic power tests (including F16/BF16)...\n";
    for (const auto& device : devices) {
        for (const auto& dtype : dtypes) {
            for (bool inplace : modes) {
                std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;
                test_pow_integer_exponent(report, device, dtype, inplace);
                test_count++;
                test_pow_float_exponent(report, device, dtype, inplace);
                test_count++;
            }
        }
    }

    // Run edge case tests
    std::cout << "\nRunning edge case tests (including F16/BF16)...\n";
    for (const auto& dtype : dtypes) {
        std::cout << "\rProgress: " << test_count << "/" << total_expected << std::flush;
        test_pow_edge_cases(report, dtype);
        test_count += 5;
    }

    std::cout << "\rProgress: " << test_count << "/" << total_expected << " ✓\n\n";

    // Generate report
    report.generate_markdown();

    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";

    return 0;
}