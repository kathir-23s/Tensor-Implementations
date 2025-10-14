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
#define COLOR_RED     "\033[31m"

// ========================================================================
// SYSTEM INFO DETECTION
// ========================================================================

void print_system_info() {
    std::cout << COLOR_CYAN << "\n========================================\n";
    std::cout << "SYSTEM CONFIGURATION\n";
    std::cout << "========================================\n" << COLOR_RESET;
    
    #ifdef _OPENMP
    std::cout << "OpenMP:           " << COLOR_GREEN << "ENABLED" << COLOR_RESET 
              << " (" << omp_get_max_threads() << " threads)\n";
    #else
    std::cout << "OpenMP:           " << COLOR_YELLOW << "DISABLED" << COLOR_RESET << "\n";
    #endif
    
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
    
    std::cout << "\n";
}

// ========================================================================
// BENCHMARK UTILITIES
// ========================================================================

struct BenchmarkResult {
    std::string name;
    std::string shape_info;
    std::string reduction_axes;
    double time_ms;
    double throughput_gflops;
    size_t input_elements;
    size_t output_elements;
    bool keepdim;
};

std::vector<BenchmarkResult> results;

void print_benchmark_header() {
    std::cout << COLOR_BOLD << "\n========================================\n";
    std::cout << "PARTIAL REDUCTION PERFORMANCE BENCHMARKS\n";
    std::cout << "========================================\n" << COLOR_RESET;
    std::cout << std::left << std::setw(35) << "Test" 
              << std::setw(20) << "Input Shape"
              << std::setw(12) << "Reduce Axes"
              << std::right << std::setw(10) << "Time(ms)" 
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "In/Out MB\n";
    std::cout << std::string(101, '-') << "\n";
}

void print_benchmark_result(const BenchmarkResult& result) {
    std::cout << std::left << std::setw(35) << result.name;
    std::cout << std::setw(20) << result.shape_info;
    std::cout << std::setw(12) << result.reduction_axes;
    std::cout << std::right << COLOR_GREEN << std::setw(10) << std::fixed 
              << std::setprecision(2) << result.time_ms << COLOR_RESET;
    
    if (result.throughput_gflops > 0) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) 
                  << result.throughput_gflops;
    } else {
        std::cout << std::setw(12) << "-";
    }
    
    double input_mb = (result.input_elements * 4) / (1024.0 * 1024.0); // assuming 4 bytes
    double output_mb = (result.output_elements * 4) / (1024.0 * 1024.0);
    std::cout << std::setw(6) << std::fixed << std::setprecision(1) << input_mb 
              << "/" << std::setw(5) << std::fixed << std::setprecision(1) << output_mb << "\n";
}

// ========================================================================
// BENCHMARK 1: 2D MATRIX REDUCTIONS (Row vs Column)
// ========================================================================

void benchmark_2d_reductions() {
    std::cout << COLOR_CYAN << "\n=== Test 1: 2D Matrix Reductions ===" << COLOR_RESET << "\n";
    
    const int64_t rows = 10000;
    const int64_t cols = 10000;
    size_t total = rows * cols;
    
    // Reduce along axis 0 (column-wise, output shape: [cols])
    {
        Tensor t(Shape{{rows, cols}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.5f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum FP32 (row reduction)",
            "[10K, 10K]",
            "axis=0",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(cols),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce along axis 1 (row-wise, output shape: [rows])
    {
        Tensor t(Shape{{rows, cols}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.5f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum FP32 (col reduction)",
            "[10K, 10K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(rows),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Mean along axis 0 with keepdim
    {
        Tensor t(Shape{{rows, cols}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 2.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(t, {0}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0); // sum + divide
        
        results.push_back({
            "Mean FP32 (keepdim=true)",
            "[10K, 10K]",
            "axis=0",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(cols),
            true
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 2: 3D TENSOR REDUCTIONS (Various Axis Patterns)
// ========================================================================

void benchmark_3d_reductions() {
    std::cout << COLOR_CYAN << "\n=== Test 2: 3D Tensor Reductions ===" << COLOR_RESET << "\n";
    
    const int64_t dim0 = 128;
    const int64_t dim1 = 512;
    const int64_t dim2 = 512;
    size_t total = dim0 * dim1 * dim2;
    
    // Reduce along last axis (most contiguous)
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {2}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum 3D (last axis)",
            "[128,512,512]",
            "axis=2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim0 * dim1),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce along middle axis
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum 3D (middle axis)",
            "[128,512,512]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim0 * dim2),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce along first axis (least contiguous)
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum 3D (first axis)",
            "[128,512,512]",
            "axis=0",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim1 * dim2),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce along multiple axes
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {0, 2}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum 3D (multi-axis)",
            "[128,512,512]",
            "axes=0,2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim1),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 3: LLM ATTENTION PATTERNS (4D Tensors)
// ========================================================================

void benchmark_llm_attention_patterns() {
    std::cout << COLOR_CYAN << "\n=== Test 3: LLM Attention Patterns (4D) ===" << COLOR_RESET << "\n";
    
    // LLaMA-7B dimensions: [batch, heads, seq_len, d_k]
    const int64_t batch = 8;
    const int64_t heads = 32;
    const int64_t seq = 2048;
    const int64_t d_k = 128;
    size_t total = batch * heads * seq * d_k;
    
    // Reduce across d_k (typical for attention score normalization)
    {
        Tensor attention(Shape{{batch, heads, seq, d_k}}, 
                        TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(total, bfloat16_t(0.5f));
        attention.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(attention, {3}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Attention Sum BF16",
            "[8,32,2048,128]",
            "axis=3",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * heads * seq),
            true
        });
        print_benchmark_result(results.back());
    }
    
    // Max across sequence (for attention masking)
    {
        Tensor attention(Shape{{batch, heads, seq, d_k}}, 
                        TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data(total, float16_t(0.3f));
        attention.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_max(attention, {2}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Attention Max FP16",
            "[8,32,2048,128]",
            "axis=2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * heads * d_k),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Mean across batch and heads (global attention stats)
    {
        Tensor attention(Shape{{batch, heads, seq, d_k}}, 
                        TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 0.4f);
        attention.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(attention, {0, 1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Attention Mean FP32",
            "[8,32,2048,128]",
            "axes=0,1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(seq * d_k),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 4: LAYER NORMALIZATION PATTERNS
// ========================================================================

void benchmark_layernorm_patterns() {
    std::cout << COLOR_CYAN << "\n=== Test 4: Layer Normalization Patterns ===" << COLOR_RESET << "\n";
    
    // [batch, seq_len, hidden_dim]
    const int64_t batch = 32;
    const int64_t seq = 512;
    const int64_t hidden = 4096;
    size_t total = batch * seq * hidden;
    
    // Mean across hidden dimension (step 1 of LayerNorm)
    {
        Tensor activations(Shape{{batch, seq, hidden}}, 
                          TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(total, bfloat16_t(0.02f));
        activations.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(activations, {2}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "LayerNorm Mean BF16",
            "[32,512,4096]",
            "axis=2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * seq),
            true
        });
        print_benchmark_result(results.back());
    }
    
    // Sum for variance calculation
    {
        Tensor squared_diff(Shape{{batch, seq, hidden}}, 
                           TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 0.01f);
        squared_diff.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(squared_diff, {2}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "LayerNorm Var FP32",
            "[32,512,4096]",
            "axis=2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * seq),
            true
        });
        print_benchmark_result(results.back());
    }
    
    // Larger batch for training
    const int64_t large_batch = 128;
    size_t large_total = large_batch * seq * hidden;
    
    {
        Tensor activations(Shape{{large_batch, seq, hidden}}, 
                          TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data(large_total, float16_t(0.015f));
        activations.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(activations, {2}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * large_total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "LayerNorm Large Batch FP16",
            "[128,512,4096]",
            "axis=2",
            time_ms,
            gflops,
            large_total,
            static_cast<size_t>(large_batch * seq),
            true
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 5: BATCH STATISTICS (Reduce across batch)
// ========================================================================

void benchmark_batch_statistics() {
    std::cout << COLOR_CYAN << "\n=== Test 5: Batch Statistics Patterns ===" << COLOR_RESET << "\n";
    
    // Reduce across batch dimension
    const int64_t batch = 256;
    const int64_t channels = 512;
    const int64_t height = 28;
    const int64_t width = 28;
    size_t total = batch * channels * height * width;
    
    // Mean across batch (batch normalization)
    {
        Tensor features(Shape{{batch, channels, height, width}}, 
                       TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 0.5f);
        features.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(features, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "BatchNorm Mean FP32",
            "[256,512,28,28]",
            "axis=0",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(channels * height * width),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce across spatial dimensions
    {
        Tensor features(Shape{{batch, channels, height, width}}, 
                       TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 0.3f);
        features.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(features, {2, 3}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Global Avg Pool FP32",
            "[256,512,28,28]",
            "axes=2,3",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * channels),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 6: INDEX REDUCTIONS (ArgMax/ArgMin)
// ========================================================================

void benchmark_index_reductions() {
    std::cout << COLOR_CYAN << "\n=== Test 6: Index Reduction Patterns ===" << COLOR_RESET << "\n";
    
    // ArgMax across vocabulary (beam search)
    const int64_t batch = 32;
    const int64_t seq = 512;
    const int64_t vocab = 50000;
    size_t total = batch * seq * vocab;
    
    {
        Tensor logits(Shape{{batch, seq, vocab}}, 
                     TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total);
        for (size_t i = 0; i < total; i++) {
            data[i] = static_cast<float>(i % 1000) / 1000.0f;
        }
        logits.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_argmax(logits, {2}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "ArgMax Vocab FP32",
            "[32,512,50K]",
            "axis=2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * seq),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // ArgMin across sequence
    {
        Tensor attention_scores(Shape{{batch, seq, seq}}, 
                               TensorOptions().with_dtype(Dtype::Float16));
        size_t attn_total = batch * seq * seq;
        std::vector<float16_t> data(attn_total, float16_t(0.5f));
        attention_scores.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_argmin(attention_scores, {2}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (attn_total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "ArgMin Attention FP16",
            "[32,512,512]",
            "axis=2",
            time_ms,
            gflops,
            attn_total,
            static_cast<size_t>(batch * seq),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 7: MIXED PRECISION PARTIAL REDUCTIONS
// ========================================================================

void benchmark_mixed_precision_partial() {
    std::cout << COLOR_CYAN << "\n=== Test 7: Mixed Precision Comparisons ===" << COLOR_RESET << "\n";
    
    const int64_t dim0 = 256;
    const int64_t dim1 = 1024;
    const int64_t dim2 = 1024;
    size_t total = dim0 * dim1 * dim2;
    
    // FP32
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.5f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Partial Mean FP32",
            "[256,1K,1K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim0 * dim2),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // FP16
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Float16));
        std::vector<float16_t> data(total, float16_t(1.5f));
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Partial Mean FP16",
            "[256,1K,1K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim0 * dim2),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // BF16
    {
        Tensor t(Shape{{dim0, dim1, dim2}}, TensorOptions().with_dtype(Dtype::Bfloat16));
        std::vector<bfloat16_t> data(total, bfloat16_t(1.5f));
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_mean(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Partial Mean BF16",
            "[256,1K,1K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(dim0 * dim2),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 8: KEEPDIM vs NO KEEPDIM (Impact on Performance)
// ========================================================================

void benchmark_keepdim_impact() {
    std::cout << COLOR_CYAN << "\n=== Test 8: KeepDim Impact on Performance ===" << COLOR_RESET << "\n";
    
    const int64_t batch = 64;
    const int64_t seq = 1024;
    const int64_t hidden = 2048;
    size_t total = batch * seq * hidden;
    
    // Without keepdim
    {
        Tensor t(Shape{{batch, seq, hidden}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum (keepdim=false)",
            "[64,1K,2K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * hidden),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // With keepdim
    {
        Tensor t(Shape{{batch, seq, hidden}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum (keepdim=true)",
            "[64,1K,2K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * hidden),
            true
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 9: CONTIGUOUS vs NON-CONTIGUOUS AXIS REDUCTION
// ========================================================================

void benchmark_axis_contiguity() {
    std::cout << COLOR_CYAN << "\n=== Test 9: Memory Layout Impact ===" << COLOR_RESET << "\n";
    
    const int64_t d0 = 100, d1 = 500, d2 = 500, d3 = 8;
    size_t total = d0 * d1 * d2 * d3;
    
    // Reduce last axis (most contiguous in memory)
    {
        Tensor t(Shape{{d0, d1, d2, d3}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {3}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum Last Axis (contiguous)",
            "[100,500,500,8]",
            "axis=3",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(d0 * d1 * d2),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce first axis (least contiguous)
    {
        Tensor t(Shape{{d0, d1, d2, d3}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum First Axis (strided)",
            "[100,500,500,8]",
            "axis=0",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(d1 * d2 * d3),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Reduce middle axes
    {
        Tensor t(Shape{{d0, d1, d2, d3}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1, 2}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum Middle Axes (mixed)",
            "[100,500,500,8]",
            "axes=1,2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(d0 * d3),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 10: EXTREME SHAPES (Very Large vs Very Small Dimensions)
// ========================================================================

void benchmark_extreme_shapes() {
    std::cout << COLOR_CYAN << "\n=== Test 10: Extreme Shape Patterns ===" << COLOR_RESET << "\n";
    
    // Very wide matrix (small height, large width)
    {
        const int64_t rows = 128;
        const int64_t cols = 100000;
        size_t total = rows * cols;
        
        Tensor t(Shape{{rows, cols}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {0}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Wide Matrix Reduce",
            "[128,100K]",
            "axis=0",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(cols),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Very tall matrix (large height, small width)
    {
        const int64_t rows = 100000;
        const int64_t cols = 128;
        size_t total = rows * cols;
        
        Tensor t(Shape{{rows, cols}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Tall Matrix Reduce",
            "[100K,128]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(rows),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // Many small dimensions
    {
        const int64_t d0 = 16, d1 = 16, d2 = 16, d3 = 16, d4 = 16;
        size_t total = d0 * d1 * d2 * d3 * d4;
        
        Tensor t(Shape{{d0, d1, d2, d3, d4}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {2, 4}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "5D Tensor Reduce",
            "[16,16,16,16,16]",
            "axes=2,4",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(d0 * d1 * d3),
            false
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// BENCHMARK 11: NAN-AWARE PARTIAL REDUCTIONS
// ========================================================================

void benchmark_nan_aware_reductions() {
    std::cout << COLOR_CYAN << "\n=== Test 11: NaN-Aware Partial Reductions ===" << COLOR_RESET << "\n";
    
    const int64_t batch = 64;
    const int64_t seq = 512;
    const int64_t features = 1024;
    size_t total = batch * seq * features;
    
    // Regular sum (with NaN propagation)
    {
        Tensor t(Shape{{batch, seq, features}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        // Inject some NaNs
        for (size_t i = 0; i < total; i += 1000) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
        }
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_sum(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "Sum with NaN (propagate)",
            "[64,512,1K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * features),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // NaN-aware sum (skip NaNs)
    {
        Tensor t(Shape{{batch, seq, features}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 1.0f);
        for (size_t i = 0; i < total; i += 1000) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
        }
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_nansum(t, {1}, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "NanSum (skip NaNs)",
            "[64,512,1K]",
            "axis=1",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * features),
            false
        });
        print_benchmark_result(results.back());
    }
    
    // NaN-aware mean
    {
        Tensor t(Shape{{batch, seq, features}}, TensorOptions().with_dtype(Dtype::Float32));
        std::vector<float> data(total, 2.0f);
        for (size_t i = 0; i < total; i += 500) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
        }
        t.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = reduce_nanmean(t, {2}, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * total / 1e9) / (time_ms / 1000.0);
        
        results.push_back({
            "NanMean (keepdim=true)",
            "[64,512,1K]",
            "axis=2",
            time_ms,
            gflops,
            total,
            static_cast<size_t>(batch * seq),
            true
        });
        print_benchmark_result(results.back());
    }
}

// ========================================================================
// FINAL SUMMARY AND ANALYSIS
// ========================================================================

void print_summary() {
    std::cout << COLOR_MAGENTA << "\n========================================\n";
    std::cout << "PERFORMANCE SUMMARY & ANALYSIS\n";
    std::cout << "========================================\n" << COLOR_RESET;
    
    // Find best and worst throughput
    double max_gflops = 0;
    double min_gflops = 1e9;
    std::string best_test = "";
    std::string worst_test = "";
    
    for (const auto& result : results) {
        if (result.throughput_gflops > 0) {
            if (result.throughput_gflops > max_gflops) {
                max_gflops = result.throughput_gflops;
                best_test = result.name;
            }
            if (result.throughput_gflops < min_gflops) {
                min_gflops = result.throughput_gflops;
                worst_test = result.name;
            }
        }
    }
    
    std::cout << "\nThroughput Statistics:\n";
    std::cout << "  Peak:           " << COLOR_GREEN << std::fixed << std::setprecision(2) 
              << max_gflops << " GFLOPS" << COLOR_RESET << " (" << best_test << ")\n";
    std::cout << "  Slowest:        " << COLOR_YELLOW << std::fixed << std::setprecision(2) 
              << min_gflops << " GFLOPS" << COLOR_RESET << " (" << worst_test << ")\n";
    
    // Calculate total time and elements
    double total_time = 0;
    size_t total_elements = 0;
    for (const auto& result : results) {
        total_time += result.time_ms;
        total_elements += result.input_elements;
    }
    
    std::cout << "\nTotal Statistics:\n";
    std::cout << "  Total Tests:    " << results.size() << "\n";
    std::cout << "  Total Time:     " << std::fixed << std::setprecision(2) 
              << total_time / 1000.0 << " seconds\n";
    std::cout << "  Total Elements: " << std::fixed << std::setprecision(2)
              << total_elements / 1e9 << " billion\n";
    std::cout << "  Avg Throughput: " << std::fixed << std::setprecision(2)
              << (total_elements / 1e9) / (total_time / 1000.0) << " GFLOPS\n";
    
    // Performance insights
    std::cout << "\n" << COLOR_CYAN << "Key Performance Insights:\n" << COLOR_RESET;
    
    // Axis contiguity analysis
    std::cout << "\n1. Memory Layout Impact:\n";
    std::cout << "   - Last axis reductions (contiguous): " << COLOR_GREEN << "FASTEST" << COLOR_RESET << "\n";
    std::cout << "   - First axis reductions (strided): " << COLOR_YELLOW << "SLOWER" << COLOR_RESET << "\n";
    std::cout << "   - Middle/multi-axis: Variable performance\n";
    
    // Dtype comparison
    std::cout << "\n2. Data Type Performance:\n";
    std::cout << "   - FP16/BF16: " << COLOR_GREEN << "2x memory bandwidth advantage" << COLOR_RESET << "\n";
    std::cout << "   - FP32: Best precision, moderate speed\n";
    std::cout << "   - BF16: Good balance for LLM workloads\n";
    
    // Operation complexity
    std::cout << "\n3. Operation Complexity:\n";
    std::cout << "   - Sum/Max: Simple, fast operations\n";
    std::cout << "   - Mean: Sum + division (2x work)\n";
    std::cout << "   - ArgMax/ArgMin: Index tracking overhead\n";
    std::cout << "   - NaN-aware: Additional NaN checks\n";
    
    // Recommendations
    std::cout << "\n" << COLOR_MAGENTA << "Optimization Recommendations:\n" << COLOR_RESET;
    
    if (max_gflops > 50) {
        std::cout << COLOR_GREEN << "  ✓ EXCELLENT PERFORMANCE" << COLOR_RESET << "\n";
        std::cout << "    - SIMD vectorization working well\n";
        std::cout << "    - OpenMP parallelism effective\n";
        std::cout << "    - Ready for production LLM inference\n";
    } else if (max_gflops > 20) {
        std::cout << COLOR_YELLOW << "  ⚠ GOOD PERFORMANCE" << COLOR_RESET << "\n";
        std::cout << "    - Consider: -O3 -march=native flags\n";
        std::cout << "    - Profile cache utilization\n";
        std::cout << "    - Check memory bandwidth bottlenecks\n";
    } else {
        std::cout << COLOR_RED << "  ⚠ NEEDS OPTIMIZATION" << COLOR_RESET << "\n";
        std::cout << "    - Enable compiler optimizations (-O3)\n";
        std::cout << "    - Enable OpenMP if disabled\n";
        std::cout << "    - Consider architecture-specific flags\n";
    }
    
    // Memory efficiency
    double avg_reduction_ratio = 0;
    int count = 0;
    for (const auto& result : results) {
        if (result.output_elements > 0) {
            avg_reduction_ratio += static_cast<double>(result.input_elements) / result.output_elements;
            count++;
        }
    }
    avg_reduction_ratio /= count;
    
    std::cout << "\n" << COLOR_CYAN << "Memory Efficiency:\n" << COLOR_RESET;
    std::cout << "  Avg Reduction Ratio: " << std::fixed << std::setprecision(1) 
              << avg_reduction_ratio << "x\n";
    std::cout << "  (Higher = more data compression)\n";
    
    std::cout << "\n";
}

// ========================================================================
// MAIN
// ========================================================================

int main() {
    std::cout << COLOR_CYAN << COLOR_BOLD << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║    OwnTensor Partial Reduction Performance Suite          ║\n";
    std::cout << "║    Testing Multi-Dimensional Reduction Patterns           ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << COLOR_RESET << "\n";
    
    print_system_info();
    print_benchmark_header();
    
    try {
        // Run all benchmark suites
        benchmark_2d_reductions();
        benchmark_3d_reductions();
        benchmark_llm_attention_patterns();
        benchmark_layernorm_patterns();
        benchmark_batch_statistics();
        benchmark_index_reductions();
        benchmark_mixed_precision_partial();
        benchmark_keepdim_impact();
        benchmark_axis_contiguity();
        benchmark_extreme_shapes();
        benchmark_nan_aware_reductions();
        
        // Print comprehensive summary
        print_summary();
        
    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "\nERROR: " << e.what() << COLOR_RESET << "\n";
        return 1;
    }
    
    std::cout << COLOR_GREEN << "✓ All partial reduction benchmarks completed successfully!\n" 
              << COLOR_RESET << "\n";
    
    return 0;
}