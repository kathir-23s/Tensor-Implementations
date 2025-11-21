#include "core/Tensor.h"
#include "ops/helpers/GenMatmulUtils.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace OwnTensor;

struct BenchmarkResult {
    double min_ms;
    double max_ms;
    double avg_ms;
};

BenchmarkResult run_benchmark(int64_t N, int iterations, bool use_optimized) {
    Shape shape = {{N, N}};
    // Pre-allocate tensors to avoid allocation noise in the loop
    Tensor A = Tensor::randn(shape, TensorOptions());
    Tensor B = Tensor::randn(shape, TensorOptions());
    Tensor C = Tensor::zeros(shape, TensorOptions());

    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        if (use_optimized) {
            cpu_matmul_optimized(A, B, C);
        } else {
            cpu_matmul_general(A, B, C);
        }
    }

    // Actual Benchmark
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (use_optimized) {
            cpu_matmul_optimized(A, B, C);
        } else {
            cpu_matmul_general(A, B, C);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double sum_t = 0.0;
    for (double t : times) sum_t += t;
    double avg_t = sum_t / times.size();

    return {min_t, max_t, avg_t};
}

int main() {
    std::vector<int64_t> sizes = {10, 32, 64, 128, 256, 512, 1024};
    int iterations = 5;

    std::cout << "====================================================================================================\n";
    std::cout << "                                  Matmul Benchmark (5 runs)                                         \n";
    std::cout << "====================================================================================================\n";
    std::cout << std::setw(10) << "Size" 
              << " | " << std::setw(25) << "General (Min/Avg/Max)" 
              << " | " << std::setw(25) << "Optimized (Min/Avg/Max)" 
              << " | " << std::setw(10) << "Speedup (Avg)" << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    for (int64_t N : sizes) {
        BenchmarkResult res_gen = run_benchmark(N, iterations, false);
        BenchmarkResult res_opt = run_benchmark(N, iterations, true);

        double speedup = res_gen.avg_ms / res_opt.avg_ms;

        std::cout << std::setw(10) << (std::to_string(N) + "x" + std::to_string(N))
                  << " | " 
                  << std::fixed << std::setprecision(2) 
                  << std::setw(7) << res_gen.min_ms << "/" << std::setw(7) << res_gen.avg_ms << "/" << std::setw(7) << res_gen.max_ms
                  << " | " 
                  << std::setw(7) << res_opt.min_ms << "/" << std::setw(7) << res_opt.avg_ms << "/" << std::setw(7) << res_opt.max_ms
                  << " | " 
                  << std::setw(9) << speedup << "x" << "\n";
    }
    std::cout << "====================================================================================================\n";

    return 0;
}
