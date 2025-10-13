//full reduction tests implementing data+threadlevel parallelism 
#include "Tensor.h"
#include "Reduction.h"
#include "Types.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace OwnTensor;

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_BOLD    "\033[1m"

// ========================================================================
// SYSTEM INFO DETECTION
// ========================================================================

void print_system_info() {
    std::cout << COLOR_CYAN << "\n========================================\n";
    std::cout << "SYSTEM CONFIGURATION\n";
    std::cout << "========================================\n" << COLOR_RESET;
    
    // OpenMP thread count
    #ifdef _OPENMP
    std::cout << "OpenMP:           " << COLOR_GREEN << "ENABLED" << COLOR_RESET 
              << " (" << omp_get_max_threads() << " threads)\n";
    #else
    std::cout << "OpenMP:           " << COLOR_YELLOW << "DISABLED" << COLOR_RESET << "\n";
    #endif
    
    // Compiler optimizations
    #ifdef __OPTIMIZE__
    std::cout << "Optimizations:    " << COLOR_GREEN << "ENABLED" << COLOR_RESET;
    #else
    std::cout << "Optimizations:    " << COLOR_YELLOW << "DISABLED" << COLOR_RESET;
    #endif
    
    #ifdef __OPTIMIZE__
        #if __OPTIMIZE__ >= 3
        std::cout << " (-O3)\n";
        #elif __OPTIMIZE__ >= 2
        std::cout << " (-O2)\n";
        #else
        std::cout << " (-O1)\n";
        #endif
    #else
    std::cout << " (-O0)\n";
    #endif
    
    // CPU Architecture
    std::cout << "Target CPU:       ";
    #ifdef __AVX512F__
    std::cout << COLOR_GREEN << "AVX512" << COLOR_RESET << " (16 floats/cycle)\n";
    #elif defined(__AVX2__)
    std::cout << COLOR_GREEN << "AVX2" << COLOR_RESET << " (8 floats/cycle)\n";
    #elif defined(__AVX__)
    std::cout << COLOR_GREEN << "AVX" << COLOR_RESET << " (8 floats/cycle)\n";
    #elif defined(__SSE4_2__)
    std::cout << COLOR_YELLOW << "SSE4.2" << COLOR_RESET << " (4 floats/cycle)\n";
    #else
    std::cout << COLOR_YELLOW << "Generic" << COLOR_RESET << " (1 float/cycle)\n";
    #endif
    
    // Native architecture
    #ifdef __MARCH__
    std::cout << "Native Arch:      " << COLOR_GREEN << "YES" << COLOR_RESET << " (-march=native)\n";
    #else
    std::cout << "Native Arch:      " << COLOR_YELLOW << "NO" << COLOR_RESET << "\n";
    #endif
    
    std::cout << "\n";
}

// ========================================================================
// BENCHMARK UTILITIES
// ========================================================================

struct BenchmarkResult {
    std::string name;
    double time_ms;
    double throughput_gflops;
    size_t elements;
};

std::vector<BenchmarkResult> results;

void print_benchmark_header() {
    std::cout << COLOR_BOLD << "\n========================================\n";
    std::cout << "PERFORMANCE BENCHMARKS\n";
    std::cout << "========================================\n" << COLOR_RESET;
    std::cout << std::left << std::setw(40) << "Test" 
              << std::right << std::setw(12) << "Time (ms)" 
              << std::setw(15) << "Throughput" 
              << std::setw(15) << "Elements\n";
    std::cout << std::string(82, '-') << "\n";
}

void print_benchmark_result(const BenchmarkResult& result) {
    std::cout << std::left << std::setw(40) << result.name;
    std::cout << std::right << COLOR_GREEN << std::setw(12) << std::fixed 
              << std::setprecision(2) << result.time_ms << COLOR_RESET;
    
    if (result.throughput_gflops > 0) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) 
                  << result.throughput_gflops << " GFLOPS";
    } else {
        std::cout << std::setw(15) << "-";
    }
    
    std::cout << std::setw(12) << result.elements / 1e6 << "M\n";
}

// ========================================================================
// BENCHMARK 1: MASSIVE SUM REDUCTION (50M ELEMENTS)
// ========================================================================

void benchmark_massive_sum() {
    std::cout << COLOR_CYAN << "\nTest 1: Massive Sum Reduction" << COLOR_RESET << "\n";
    
    const int64_t SIZE = 50000000; // 50 million
    
    // Float32
    {
        Tensor t(Shape{{SIZE}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(SIZE, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        // 1 addition per element = 50M FLOPS
        double gflops = (SIZE / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Sum Float32 (50M)", time_ms, gflops, static_cast<size_t>(SIZE)});
        print_benchmark_result(results.back());
    }
    
    // Float16
    {
        Tensor t(Shape{{SIZE}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data(SIZE, float16_t(1.0f));
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (SIZE / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Sum Float16 (50M)", time_ms, gflops, static_cast<size_t>(SIZE)});
        print_benchmark_result(results.back());
    }
    
    // Bfloat16
    {
        Tensor t(Shape{{SIZE}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(SIZE, bfloat16_t(1.0f));
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (SIZE / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Sum Bfloat16 (50M)", time_ms, gflops, static_cast<size_t>(SIZE)});
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 2: LLM ATTENTION PATTERN (Multi-dimensional reduction)
// ========================================================================

void benchmark_attention_pattern() {
    std::cout << COLOR_CYAN << "\nTest 2: LLM Attention Pattern" << COLOR_RESET << "\n";
    
    // Typical LLaMA-7B attention: [batch=8, heads=32, seq=2048, d_k=128]
    const int64_t batch = 8;
    const int64_t heads = 32;
    const int64_t seq = 2048;
    const int64_t d_k = 128;
    
    size_t total_elements = batch * heads * seq * d_k;
    
    {
        Tensor attention(Shape{{batch, heads, seq, d_k}}, 
                        TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(total_elements, bfloat16_t(0.5f));
        attention.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(attention, {3}, true); // Sum across d_k
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total_elements / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Attention Sum (BF16, 64M)", time_ms, gflops, total_elements});
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 3: LAYER NORMALIZATION (Mean across hidden dim)
// ========================================================================

void benchmark_layernorm() {
    std::cout << COLOR_CYAN << "\nTest 3: Layer Normalization Pattern" << COLOR_RESET << "\n";
    
    // [batch=32, seq=512, hidden=4096] - LLaMA-7B size
    const int64_t batch = 32;
    const int64_t seq = 512;
    const int64_t hidden = 4096;
    
    size_t total_elements = batch * seq * hidden;
    
    {
        Tensor activations(Shape{{batch, seq, hidden}}, 
                          TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(total_elements, bfloat16_t(0.01f));
        activations.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(activations, {2}, true); // Mean across hidden
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        // Mean = sum + divide, ~2x operations
        double gflops = (2.0 * total_elements / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"LayerNorm Mean (BF16, 64M)", time_ms, gflops, total_elements});
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 4: GRADIENT ACCUMULATION
// ========================================================================

void benchmark_gradient_accumulation() {
    std::cout << COLOR_CYAN << "\nTest 4: Gradient Accumulation Pattern" << COLOR_RESET << "\n";
    
    // [microbatches=16, params=20M] - Simulating 7B model chunk
    const int64_t microbatches = 16;
    const int64_t params = 20000000;
    
    size_t total_elements = microbatches * params;
    
    {
        Tensor gradients(Shape{{microbatches, params}}, 
                        TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total_elements, 0.0001f);
        gradients.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(gradients, {0}, false); // Sum across microbatches
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total_elements / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Gradient Accum (FP32, 320M)", time_ms, gflops, total_elements});
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 5: ARGMAX (Index Reduction)
// ========================================================================

void benchmark_argmax() {
    std::cout << COLOR_CYAN << "\nTest 5: ArgMax (Beam Search Pattern)" << COLOR_RESET << "\n";
    
    const int64_t batch = 32;
    const int64_t seq = 512;
    const int64_t vocab = 50000;
    
    size_t total_elements = batch * seq * vocab;
    
    {
        Tensor logits(Shape{{batch, seq, vocab}}, 
                     TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total_elements);
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = static_cast<float>(i % 1000) / 1000.0f;
        }
        logits.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_argmax(logits, {2}, false); // ArgMax across vocab
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        // ArgMax = compare operations
        double gflops = (total_elements / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"ArgMax (FP32, 819M)", time_ms, gflops, total_elements});
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 6: MIXED PRECISION COMPARISON
// ========================================================================

void benchmark_mixed_precision() {
    std::cout << COLOR_CYAN << "\nTest 6: Mixed Precision Comparison" << COLOR_RESET << "\n";
    
    const int64_t SIZE = 100000000; // 100M elements
    
    // FP32 baseline
    {
        Tensor t(Shape{{SIZE}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(SIZE, 1.5f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (SIZE / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Mixed: FP32 (100M)", time_ms, gflops, static_cast<size_t>(SIZE)});
        print_benchmark_result(results.back());
    }
    
    // FP16 (should be faster due to memory bandwidth)
    {
        Tensor t(Shape{{SIZE}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data(SIZE, float16_t(1.5f));
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (SIZE / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Mixed: FP16 (100M)", time_ms, gflops, static_cast<size_t>(SIZE)});
        print_benchmark_result(results.back());
    }
    
    // BF16
    {
        Tensor t(Shape{{SIZE}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(SIZE, bfloat16_t(1.5f));
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (SIZE / 1e9) / (time_ms / 1000.0);
        
        results.push_back({"Mixed: BF16 (100M)", time_ms, gflops, static_cast<size_t>(SIZE)});
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// FINAL SUMMARY
// ========================================================================

void print_summary() {
    std::cout << COLOR_MAGENTA << "\n========================================\n";
    std::cout << "PERFORMANCE SUMMARY\n";
    std::cout << "========================================\n" << COLOR_RESET;
    
    // Find best throughput
    double max_gflops = 0;
    std::string best_test = "";
    for (const auto& result : results) {
        if (result.throughput_gflops > max_gflops) {
            max_gflops = result.throughput_gflops;
            best_test = result.name;
        }
    }
    
    std::cout << "Peak Throughput:  " << COLOR_GREEN << std::fixed 
              << std::setprecision(2) << max_gflops << " GFLOPS" 
              << COLOR_RESET << " (" << best_test << ")\n";
    
    // Calculate total time
    double total_time = 0;
    size_t total_elements = 0;
    for (const auto& result : results) {
        total_time += result.time_ms;
        total_elements += result.elements;
    }
    
    std::cout << "Total Time:       " << std::fixed << std::setprecision(2) 
              << total_time / 1000.0 << " seconds\n";
    std::cout << "Total Elements:   " << total_elements / 1e9 << " billion\n";
    std::cout << "Avg Throughput:   " << (total_elements / 1e9) / (total_time / 1000.0) 
              << " GFLOPS\n";
    
    // Performance interpretation
    std::cout << "\n" << COLOR_CYAN << "Performance Interpretation:\n" << COLOR_RESET;
    if (max_gflops > 50) {
        std::cout << COLOR_GREEN << "âœ EXCELLENT" << COLOR_RESET 
                  << " - Fully optimized with SIMD acceleration\n";
    } else if (max_gflops > 10) {
        std::cout << COLOR_YELLOW << "âœ GOOD" << COLOR_RESET 
                  << " - OpenMP parallelism working, SIMD may be partial\n";
    } else {
        std::cout << COLOR_YELLOW << "âš ï¸  NEEDS OPTIMIZATION" << COLOR_RESET 
                  << " - Consider enabling -O3 -march=native\n";
    }
    
    std::cout << "\n";
}

// ========================================================================
// MAIN
// ========================================================================

int main() {
    std::cout << COLOR_CYAN << COLOR_BOLD << "\n";
    std::cout << "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n";
    std::cout << "â•'                                                            â•'\n";
    std::cout << "â•'     OwnTensor Performance Benchmark Suite                 â•'\n";
    std::cout << "â•'     Testing Thread + Data Level Parallelism               â•'\n";
    std::cout << "â•'                                                            â•'\n";
    std::cout << "â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n";
    std::cout << COLOR_RESET << "\n";
    
    print_system_info();
    print_benchmark_header();
    
    try {
        benchmark_massive_sum();
        benchmark_attention_pattern();
        benchmark_layernorm();
        benchmark_gradient_accumulation();
        benchmark_argmax();
        benchmark_mixed_precision();
        
        print_summary();
        
    } catch (const std::exception& e) {
        std::cout << COLOR_YELLOW << "\nERROR: " << e.what() << COLOR_RESET << "\n";
        return 1;
    }
    
    return 0;
}