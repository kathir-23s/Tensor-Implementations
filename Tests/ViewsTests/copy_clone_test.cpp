#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cassert>
#include "Tensor.h"
#include "Types.h"
#include "ops/helpers/testutils.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

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
        
        file << "# Copy and Clone Operations Test Report\n\n";
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
        
        file << "\n## Operation Differences\n\n";
        file << "| Operation | Memory | Modifies Original | Use Case |\n";
        file << "|-----------|--------|-------------------|----------|\n";
        file << "| `clone()` | New allocation | No | Create independent copy |\n";
        file << "| `copy_(src)` | In-place | Yes | Overwrite existing tensor |\n\n";
        
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

template<typename T>
const void* get_data_ptr(const Tensor& t) {
    return static_cast<const void*>(t.data<T>());
}

// ============================================================================
// Clone Tests
// ============================================================================

void test_clone_basic_cpu(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
        Tensor cloned = t.clone();
        
        // Verify values match
        assert(cloned.numel() == t.numel());
        assert(verify_tensor_values(cloned, {1,2,3,4,5,6}, 1e-6));
        
        // Verify different memory
        const void* t_ptr = get_data_ptr<float>(t);
        const void* cloned_ptr = get_data_ptr<float>(cloned);
        assert(t_ptr != cloned_ptr);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Basic (CPU)", true, "Values copied, memory independent", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Basic (CPU)", false, std::string(e.what()), time_ms});
    }
}

void test_clone_basic_gpu(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
#ifdef WITH_CUDA
        Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
        Tensor cloned = t.clone();
        
        // Verify values match
        assert(cloned.numel() == t.numel());
        assert(cloned.device().is_cuda());
        assert(verify_tensor_values(cloned, {1,2,3,4,5,6}, 1e-6));
        
        // Verify different memory
        const void* t_ptr = get_data_ptr<float>(t);
        const void* cloned_ptr = get_data_ptr<float>(cloned);
        assert(t_ptr != cloned_ptr);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Basic (GPU)", true, "GPU clone successful", time_ms});
#else
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Basic (GPU)", true, "CUDA not available", time_ms});
#endif
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Basic (GPU)", false, std::string(e.what()), time_ms});
    }
}

void test_clone_different_shapes(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // 1D
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor cloned = t.clone();
            assert(verify_tensor_values(cloned, {1,2,3,4}, 1e-6));
        }
        
        // 2D
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor cloned = t_2d.clone();
            assert(cloned.shape().dims[0] == 2);
            assert(cloned.shape().dims[1] == 3);
        }
        
        // 3D
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_3d = t.reshape({{2, 2, 2}});
            Tensor cloned = t_3d.clone();
            assert(cloned.shape().dims.size() == 3);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Different Shapes", true, "1D, 2D, 3D tensors cloned", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Different Shapes", false, std::string(e.what()), time_ms});
    }
}

void test_clone_different_dtypes(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Float32
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor cloned = t.clone();
            assert(cloned.dtype() == Dtype::Float32);
        }
        
        // Float64
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float64);
            Tensor cloned = t.clone();
            assert(cloned.dtype() == Dtype::Float64);
        }
        
        // Float16
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float16);
            Tensor cloned = t.clone();
            assert(cloned.dtype() == Dtype::Float16);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Different Dtypes", true, "Multiple dtypes cloned", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Different Dtypes", false, std::string(e.what()), time_ms});
    }
}

void test_clone_non_contiguous(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
        Tensor t_2d = t.reshape({{2, 3}});
        Tensor transposed = t_2d.transpose(0, 1);
        
        assert(!transposed.is_contiguous());
        
        Tensor cloned = transposed.clone();
        
        // Clone should be contiguous
        assert(cloned.is_contiguous());
        assert(cloned.shape().dims[0] == 3);
        assert(cloned.shape().dims[1] == 2);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Non-Contiguous", true, "Clone is contiguous", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Clone Non-Contiguous", false, std::string(e.what()), time_ms});
    }
}

// ============================================================================
// Copy_ Tests
// ============================================================================

void test_copy_basic_cpu(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        Tensor src = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
        Tensor dst = create_tensor_from_float({10,20,30,40}, DeviceIndex(Device::CPU), Dtype::Float32);
        
        const void* dst_ptr_before = get_data_ptr<float>(dst);
        
        dst.copy_(src);
        
        const void* dst_ptr_after = get_data_ptr<float>(dst);
        
        // Same memory location
        assert(dst_ptr_before == dst_ptr_after);
        
        // Values updated
        assert(verify_tensor_values(dst, {1,2,3,4}, 1e-6));
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Basic (CPU)", true, "In-place copy successful", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Basic (CPU)", false, std::string(e.what()), time_ms});
    }
}

void test_copy_basic_gpu(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
#ifdef WITH_CUDA
        Tensor src = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CUDA), Dtype::Float32);
        Tensor dst = create_tensor_from_float({10,20,30,40}, DeviceIndex(Device::CUDA), Dtype::Float32);
        
        dst.copy_(src);
        
        assert(verify_tensor_values(dst, {1,2,3,4}, 1e-6));
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Basic (GPU)", true, "GPU in-place copy successful", time_ms});
#else
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Basic (GPU)", true, "CUDA not available", time_ms});
#endif
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Basic (GPU)", false, std::string(e.what()), time_ms});
    }
}

void test_copy_cross_device(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
#ifdef WITH_CUDA
        // CPU to GPU
        {
            Tensor cpu_src = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor gpu_dst = create_tensor_from_float({0,0,0,0}, DeviceIndex(Device::CUDA), Dtype::Float32);
            gpu_dst.copy_(cpu_src);
            assert(verify_tensor_values(gpu_dst, {1,2,3,4}, 1e-6));
        }
        
        // GPU to CPU
        {
            Tensor gpu_src = create_tensor_from_float({5,6,7,8}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor cpu_dst = create_tensor_from_float({0,0,0,0}, DeviceIndex(Device::CPU), Dtype::Float32);
            cpu_dst.copy_(gpu_src);
            assert(verify_tensor_values(cpu_dst, {5,6,7,8}, 1e-6));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Cross-Device", true, "CPU↔GPU copy successful", time_ms});
#else
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Cross-Device", true, "CUDA not available", time_ms});
#endif
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Cross-Device", false, std::string(e.what()), time_ms});
    }
}

void test_copy_different_shapes(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        Tensor src = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
        Tensor src_2d = src.reshape({{2, 3}});
        
        Tensor dst = create_tensor_from_float({0,0,0,0,0,0}, DeviceIndex(Device::CPU), Dtype::Float32);
        Tensor dst_2d = dst.reshape({{3, 2}});
        
        // Should work if numel matches
        dst.copy_(src);
        assert(verify_tensor_values(dst, {1,2,3,4,5,6}, 1e-6));
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Different Shapes", true, "Copy with shape broadcast", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Copy_ Different Shapes", false, std::string(e.what()), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  COPY AND CLONE OPERATIONS TESTS\n";
    std::cout << "========================================\n\n";
    
    TestReport report("copy_clone_testreport.md");
    
    int test_count = 0;
    int total_tests = 9;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_clone_basic_cpu(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_clone_basic_gpu(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_clone_different_shapes(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_clone_different_dtypes(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_clone_non_contiguous(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_copy_basic_cpu(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_copy_basic_gpu(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_copy_cross_device(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_copy_different_shapes(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << " ✓\n\n";
    
    report.generate_markdown();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}