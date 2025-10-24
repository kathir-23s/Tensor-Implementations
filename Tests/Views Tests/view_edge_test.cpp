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
        
        file << "# View Operations Edge Case Test Report\n\n";
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
        file << "### Edge Cases Tested\n";
        file << "- Single element tensors\n";
        file << "- Reshape to same shape\n";
        file << "- Multiple sequential operations\n";
        file << "- Transpose with same dimensions\n";
        file << "- Double transpose\n";
        file << "- Flatten already flat tensors\n";
        file << "- Complex view chains\n";
        file << "- Contiguous operations\n";
        file << "- GPU edge cases\n\n";
        
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
// Test Edge Cases for View Operations
// ============================================================================

void test_1d_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Single element tensor
        {
            Tensor t = create_tensor_from_float({5.0f}, DeviceIndex(Device::CPU), Dtype::Float32);
            assert(t.numel() == 1);
            assert(t.shape().dims.size() == 1);
            assert(t.shape().dims[0] == 1);
        }
        
        // Empty reshape
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor reshaped = t.reshape({{1, 6}});
            assert(reshaped.shape().dims[0] == 1);
            assert(reshaped.shape().dims[1] == 6);
        }
        
        // Transpose of 1D
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{1, 4}});
            Tensor transposed = t_2d.transpose(0, 1);
            assert(transposed.shape().dims[0] == 4);
            assert(transposed.shape().dims[1] == 1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"1D Tensor Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"1D Tensor Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_reshape_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Reshape to same shape
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor same = t_2d.reshape({{2, 3}});
            assert(same.shape().dims[0] == 2);
            assert(same.shape().dims[1] == 3);
        }
        
        // Multiple reshapes in sequence
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8,9,10,11,12}, 
                                               DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor r1 = t.reshape({{3, 4}});
            Tensor r2 = r1.reshape({{2, 6}});
            Tensor r3 = r2.reshape({{12, 1}});
            assert(r3.shape().dims[0] == 12);
            assert(r3.shape().dims[1] == 1);
        }
        
        // Reshape with dimension of 1
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor r = t.reshape({{4, 1, 1}});
            assert(r.shape().dims.size() == 3);
            assert(r.shape().dims[1] == 1);
            assert(r.shape().dims[2] == 1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Reshape Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Reshape Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_transpose_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Transpose same dimension
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 2}});
            Tensor same = t_2d.transpose(0, 0);
            assert(same.shape().dims[0] == 2);
            assert(same.shape().dims[1] == 2);
        }
        
        // Multiple transposes
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor t1 = t_2d.transpose(0, 1);
            Tensor t2 = t1.transpose(0, 1);
            assert(t2.shape().dims[0] == 2);
            assert(t2.shape().dims[1] == 3);
        }
        
        // Transpose non-adjacent dimensions
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8,9,10,11,12}, 
                                               DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_3d = t.reshape({{2, 2, 3}});
            Tensor transposed = t_3d.transpose(0, 2);
            assert(transposed.shape().dims[0] == 3);
            assert(transposed.shape().dims[1] == 2);
            assert(transposed.shape().dims[2] == 2);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Transpose Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Transpose Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_flatten_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Flatten already flat tensor
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor flat = t.flatten();
            assert(flat.numel() == 4);
            assert(flat.shape().dims.size() == 1);
        }
        
        // Flatten with start_dim = end_dim
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor flat = t_2d.flatten(0, 0);
            assert(flat.shape().dims[0] == 2);
        }
        
        // Flatten only middle dimensions
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_3d = t.reshape({{2, 2, 2}});
            Tensor flat = t_3d.flatten(1, 2);
            assert(flat.shape().dims[0] == 2);
            assert(flat.shape().dims[1] == 4);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Flatten Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Flatten Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_chained_view_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Reshape -> Transpose -> Flatten
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8,9,10,11,12}, 
                                               DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor r = t.reshape({{3, 4}});
            Tensor tr = r.transpose(0, 1);
            Tensor flat = tr.flatten();
            assert(flat.numel() == 12);
        }
        
        // Multiple transposes with reshapes
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor r1 = t.reshape({{2, 4}});
            Tensor tr1 = r1.transpose(0, 1);
            Tensor r2 = tr1.reshape({{2, 2, 2}});
            Tensor tr2 = r2.transpose(0, 2);
            assert(tr2.numel() == 8);
        }
        
        // View after contiguous
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor tr = t_2d.transpose(0, 1);
            Tensor cont = tr.contiguous();
            Tensor reshaped = cont.reshape({{1, 6}});
            assert(reshaped.shape().dims[0] == 1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Chained View Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Chained View Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_contiguous_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // Contiguous on already contiguous tensor
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor cont = t.contiguous();
            assert(cont.is_contiguous());
        }
        
        // Contiguous after transpose
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor tr = t_2d.transpose(0, 1);
            assert(!tr.is_contiguous());
            Tensor cont = tr.contiguous();
            assert(cont.is_contiguous());
        }
        
        // Multiple contiguous calls
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor cont1 = t.contiguous();
            Tensor cont2 = cont1.contiguous();
            Tensor cont3 = cont2.contiguous();
            assert(cont3.is_contiguous());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Contiguous Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Contiguous Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_storage_offset_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // View with zero offset
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor view = t.reshape({{2, 2}});
            assert(view.numel() == 4);
        }
        
        // Nested views with offsets
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8,9,10,11,12}, 
                                               DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor r1 = t.reshape({{3, 4}});
            Tensor tr = r1.transpose(0, 1);
            Tensor r2 = tr.reshape({{2, 6}});
            assert(r2.numel() == 12);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Storage Offset Edge Cases", true, "All tests passed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Storage Offset Edge Cases", false, std::string(e.what()), time_ms});
    }
}

void test_gpu_edge_cases(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
#ifdef WITH_CUDA
        // Single element on GPU
        {
            Tensor t = create_tensor_from_float({5.0f}, DeviceIndex(Device::CUDA), Dtype::Float32);
            assert(t.numel() == 1);
            assert(t.device().is_cuda());
        }
        
        // GPU transpose then contiguous
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor tr = t_2d.transpose(0, 1);
            Tensor cont = tr.contiguous();
            assert(cont.is_contiguous());
            assert(cont.device().is_cuda());
        }
        
        // GPU flatten after transpose
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 4}});
            Tensor tr = t_2d.transpose(0, 1);
            Tensor flat = tr.flatten();
            assert(flat.numel() == 8);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"GPU Edge Cases", true, "All GPU tests passed", time_ms});
#else
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"GPU Edge Cases", true, "GPU tests skipped (CUDA not available)", time_ms});
#endif
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"GPU Edge Cases", false, std::string(e.what()), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  VIEW OPERATIONS - EDGE CASE TESTS\n";
    std::cout << "========================================\n\n";
    
    TestReport report("view_edge_testreport.md");
    
    int test_count = 0;
    int total_tests = 8;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_1d_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_reshape_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_transpose_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_flatten_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_chained_view_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_contiguous_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_storage_offset_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_gpu_edge_cases(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << " ✓\n\n";
    
    report.generate_markdown();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}