#include "TensorLib.h"
#include <chrono>

using namespace OwnTensor;

int main()
{
    Tensor A(Shape{{40000,35000}}, Dtype::Bfloat16, DeviceIndex(Device::CUDA,0), 0);
    A.fill(bfloat16_t(0.75f));
    // A.display(std::cout, 4);
    Tensor B(Shape{{40000,35000}}, Dtype::Bfloat16, DeviceIndex(Device::CUDA, 0), 0);
    B.fill(bfloat16_t(1.265f));
    {
    // B.display(std::cout, 4);
    Tensor A_plus_B(A.shape(), A.dtype(), A.device(), 0);
    auto start = std::chrono::high_resolution_clock::now();
    A_plus_B = A + B;
    // A_plus_B = A - B;
    // A_plus_B = A * B;
    // A_plus_B = A / B;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\n=== RESULTS CUDA (Element wise Addition) ===" << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    }

    {
    
    // B.display(std::cout, 4);
    Tensor A_plus_B(A.shape(), A.dtype(), A.device(), 0);
    auto start = std::chrono::high_resolution_clock::now();
    // A_plus_B = A + B;
    // A_plus_B = A - B;
    A_plus_B = A * B;
    // A_plus_B = A / B;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\n=== RESULTS CUDA (Element wise Multiplication) ===" << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    }

    {
    
    // B.display(std::cout, 4);
    Tensor A_plus_B(A.shape(), A.dtype(), A.device(), 0);
    auto start = std::chrono::high_resolution_clock::now();
    A_plus_B = A - B;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\n=== RESULTS CUDA (Element wise Subtraction) ===" << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    }

    {
    
    // B.display(std::cout, 4);
    Tensor A_plus_B(A.shape(), A.dtype(), A.device(), 0);
    auto start = std::chrono::high_resolution_clock::now();
    // A_plus_B = A + B;
    // A_plus_B = A - B;
    // A_plus_B = A * B;
    A_plus_B = A / B;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\n=== RESULTS CUDA (Element wise Division) ===" << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    }

    // {
    // Tensor A(Shape{{40000,25000}}, Dtype::Float16, Device::CPU, 0);
    // A.fill(float16_t(0.75f));
    // // A.display(std::cout, 4);
    // Tensor B(Shape{{40000,25000}}, Dtype::Float16, Device::CPU, 0);
    // B.fill(float16_t(1.265f));
    // // B.display(std::cout, 4);
    // Tensor A_plus_B(A.shape(), A.dtype(), A.device(), 0);
    // auto start = std::chrono::high_resolution_clock::now();
    // A_plus_B = A + B;
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "\n=== RESULTS CPU ===" << std::endl;
    // std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    // }


    // Tensor C = A.to_cuda();
    // Tensor D = B.to_cuda();
    // C = C + D;

    // A = C.to_cpu();
    // A.display(std::cout, 4);

    // std::cout << "Output shape correct: " << (shape_correct ? "YES" : "NO") << std::endl;
    // std::cout << "Values valid (no NaN/Inf): " << (values_valid ? "YES" : "NO") << std::endl;
    // std::cout << "Test status: " << (shape_correct && values_valid ? "PASSED" : "FAILED") << std::endl;
    
    // // Performance metrics
    // double total_operations = static_cast<double>(batch_size) * m * n * p;
    // double gflops = (total_operations / (duration.count() / 1000.0)) / 1e9;
    // std::cout << "Performance: " << gflops << " GFLOPs" << std::endl;
    
    std::cout << "===================================" << std::endl;

}