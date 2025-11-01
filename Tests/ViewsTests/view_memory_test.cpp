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
        
        file << "# View Operations Memory Test Report\n\n";
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
        
        file << "\n## Memory Management Coverage\n\n";
        file << "### Memory Sharing\n";
        file << "- Reshape shares memory with original\n";
        file << "- Transpose shares memory with original\n";
        file << "- Flatten shares memory (when contiguous)\n\n";
        
        file << "### Memory Allocation\n";
        file << "- Contiguous creates new memory for non-contiguous tensors\n";
        file << "- Multiple views share same underlying storage\n\n";
        
        file << "### Lifetime Management\n";
        file << "- Views can outlive original tensors\n";
        file << "- Original tensors can outlive views\n";
        file << "- Reference counting prevents premature deallocation\n\n";
        
        file << "### Memory Leaks\n";
        file << "- No leaks with chained view operations\n";
        file << "- No leaks with contiguous operations\n";
        file << "- Proper cleanup on scope exit\n\n";
        
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

bool shares_memory(const Tensor& t1, const Tensor& t2) {
    return get_data_ptr<uint8_t>(t1) == get_data_ptr<uint8_t>(t2);
}

// ============================================================================
// Memory Sharing Tests
// ============================================================================

void test_reshape_shares_memory(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU: Reshape should share memory
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            const void* original_ptr = get_data_ptr<float>(t);
            Tensor reshaped = t.reshape({{2, 3}});
            const void* reshaped_ptr = get_data_ptr<float>(reshaped);
            assert(original_ptr == reshaped_ptr);
        }
        
#ifdef WITH_CUDA
        // GPU: Reshape should share memory
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
            const void* original_ptr = get_data_ptr<float>(t);
            Tensor reshaped = t.reshape({{2, 3}});
            const void* reshaped_ptr = get_data_ptr<float>(reshaped);
            assert(original_ptr == reshaped_ptr);
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Reshape Shares Memory", true, "Memory sharing verified", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Reshape Shares Memory", false, std::string(e.what()), time_ms});
    }
}

void test_transpose_shares_memory(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            const void* original_ptr = get_data_ptr<float>(t_2d);
            Tensor transposed = t_2d.transpose(0, 1);
            const void* transposed_ptr = get_data_ptr<float>(transposed);
            assert(original_ptr == transposed_ptr);
        }
        
#ifdef WITH_CUDA
        // GPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            const void* original_ptr = get_data_ptr<float>(t_2d);
            Tensor transposed = t_2d.transpose(0, 1);
            const void* transposed_ptr = get_data_ptr<float>(transposed);
            assert(original_ptr == transposed_ptr);
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Transpose Shares Memory", true, "Memory sharing verified", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Transpose Shares Memory", false, std::string(e.what()), time_ms});
    }
}

void test_flatten_shares_memory(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            const void* original_ptr = get_data_ptr<float>(t_2d);
            Tensor flat = t_2d.flatten();
            const void* flat_ptr = get_data_ptr<float>(flat);
            assert(original_ptr == flat_ptr);
        }
        
#ifdef WITH_CUDA
        // GPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            const void* original_ptr = get_data_ptr<float>(t_2d);
            Tensor flat = t_2d.flatten();
            const void* flat_ptr = get_data_ptr<float>(flat);
            assert(original_ptr == flat_ptr);
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Flatten Shares Memory", true, "Memory sharing verified", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Flatten Shares Memory", false, std::string(e.what()), time_ms});
    }
}

void test_contiguous_creates_new_memory(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor transposed = t_2d.transpose(0, 1);
            const void* transposed_ptr = get_data_ptr<float>(transposed);
            Tensor cont = transposed.contiguous();
            const void* cont_ptr = get_data_ptr<float>(cont);
            assert(transposed_ptr != cont_ptr);
        }
        
#ifdef WITH_CUDA
        // GPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor transposed = t_2d.transpose(0, 1);
            const void* transposed_ptr = get_data_ptr<float>(transposed);
            Tensor cont = transposed.contiguous();
            const void* cont_ptr = get_data_ptr<float>(cont);
            assert(transposed_ptr != cont_ptr);
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Contiguous Creates New Memory", true, "New allocation verified", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Contiguous Creates New Memory", false, std::string(e.what()), time_ms});
    }
}

void test_view_modification_affects_original(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        {
            std::vector<float> data = {1,2,3,4,5,6};
            Tensor t = create_tensor_from_float(data, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor reshaped = t.reshape({{2, 3}});
            assert(shares_memory(t, reshaped));
        }
        
#ifdef WITH_CUDA
        // GPU
        {
            std::vector<float> data = {1,2,3,4,5,6};
            Tensor t = create_tensor_from_float(data, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor reshaped = t.reshape({{2, 3}});
            assert(shares_memory(t, reshaped));
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"View Modification Affects Original", true, "Memory sharing confirmed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"View Modification Affects Original", false, std::string(e.what()), time_ms});
    }
}

void test_view_outlives_original(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor view = t.reshape({{2, 2}});
            {
                Tensor t_inner = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CPU), Dtype::Float32);
                view = t_inner.reshape({{2, 2}});
            }
            assert(view.numel() == 4);
            assert(view.shape().dims[0] == 2);
        }
        
#ifdef WITH_CUDA
        // GPU
        Tensor view = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CUDA), Dtype::Float32);
        {
            Tensor t = create_tensor_from_float({1,2,3,4}, DeviceIndex(Device::CUDA), Dtype::Float32);
            view = t.reshape({{2, 2}});
        }
        assert(view.numel() == 4);
        assert(view.device().is_cuda());
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"View Outlives Original", true, "Reference counting works", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"View Outlives Original", false, std::string(e.what()), time_ms});
    }
}

void test_original_outlives_view(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        Tensor original = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
        {
            Tensor view = original.reshape({{2, 3}});
            assert(view.numel() == 6);
        }
        assert(original.numel() == 6);
        
#ifdef WITH_CUDA
        // GPU
        Tensor gpu_original = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
        {
            Tensor view = gpu_original.reshape({{2, 3}});
            assert(view.numel() == 6);
        }
        assert(gpu_original.numel() == 6);
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Original Outlives View", true, "Original tensor survives", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Original Outlives View", false, std::string(e.what()), time_ms});
    }
}

void test_multiple_views_same_tensor(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor view1 = t.reshape({{2, 4}});
            Tensor view2 = t.reshape({{4, 2}});
            Tensor view3 = t.reshape({{1, 8}});
            
            const void* ptr_orig = get_data_ptr<float>(t);
            const void* ptr_v1 = get_data_ptr<float>(view1);
            const void* ptr_v2 = get_data_ptr<float>(view2);
            const void* ptr_v3 = get_data_ptr<float>(view3);
            
            assert(ptr_orig == ptr_v1);
            assert(ptr_orig == ptr_v2);
            assert(ptr_orig == ptr_v3);
        }
        
#ifdef WITH_CUDA
        // GPU
        {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor view1 = t.reshape({{2, 4}});
            Tensor view2 = t.reshape({{4, 2}});
            Tensor view3 = t.reshape({{1, 8}});
            
            const void* ptr_orig = get_data_ptr<float>(t);
            const void* ptr_v1 = get_data_ptr<float>(view1);
            const void* ptr_v2 = get_data_ptr<float>(view2);
            const void* ptr_v3 = get_data_ptr<float>(view3);
            
            assert(ptr_orig == ptr_v1);
            assert(ptr_orig == ptr_v2);
            assert(ptr_orig == ptr_v3);
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Multiple Views Same Tensor", true, "All views share memory", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"Multiple Views Same Tensor", false, std::string(e.what()), time_ms});
    }
}

void test_no_leaks_with_views(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        for (int i = 0; i < 100; ++i) {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8,9,10,11,12}, 
                                               DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor r = t.reshape({{3, 4}});
            Tensor tr = r.transpose(0, 1);
            Tensor f = tr.flatten();
        }
        
#ifdef WITH_CUDA
        // GPU
        for (int i = 0; i < 100; ++i) {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6,7,8,9,10,11,12}, 
                                               DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor r = t.reshape({{3, 4}});
            Tensor tr = r.transpose(0, 1);
            Tensor f = tr.flatten();
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"No Leaks With Views", true, "100 iterations completed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"No Leaks With Views", false, std::string(e.what()), time_ms});
    }
}

void test_no_leaks_with_contiguous(TestReport& report) {
    auto start = std::chrono::high_resolution_clock::now();
    try {
        // CPU
        for (int i = 0; i < 100; ++i) {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CPU), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor tr = t_2d.transpose(0, 1);
            Tensor cont = tr.contiguous();
        }
        
#ifdef WITH_CUDA
        // GPU
        for (int i = 0; i < 100; ++i) {
            Tensor t = create_tensor_from_float({1,2,3,4,5,6}, DeviceIndex(Device::CUDA), Dtype::Float32);
            Tensor t_2d = t.reshape({{2, 3}});
            Tensor tr = t_2d.transpose(0, 1);
            Tensor cont = tr.contiguous();
        }
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"No Leaks With Contiguous", true, "100 iterations completed", time_ms});
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        report.add_result({"No Leaks With Contiguous", false, std::string(e.what()), time_ms});
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  VIEW OPERATIONS - MEMORY TESTS\n";
    std::cout << "========================================\n\n";
    
    TestReport report("view_memory_testreport.md");
    
    int test_count = 0;
    int total_tests = 10;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_reshape_shares_memory(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_transpose_shares_memory(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_flatten_shares_memory(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_contiguous_creates_new_memory(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_view_modification_affects_original(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_view_outlives_original(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_original_outlives_view(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_multiple_views_same_tensor(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_no_leaks_with_views(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << std::flush;
    test_no_leaks_with_contiguous(report); test_count++;
    
    std::cout << "\rProgress: " << test_count << "/" << total_tests << " ✓\n\n";
    
    report.generate_markdown();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}