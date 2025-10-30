#include <iostream>
#include <vector>
#include "TensorLib.h" // Your main library header

// Use your library's namespace
using namespace OwnTensor;

int main() {
    std::cout << "--- Simple CPU Matmul Test ---" << std::endl;

    // 1. Define the data for two matrices, A (2x3) and B (3x2)
    std::vector<float> a_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::vector<float> b_data = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    // 2. Create the Tensors on the CPU
    Tensor a_cpu(Shape{{3, 2}}, TensorOptions{Dtype::Float32, Device::CPU});
    Tensor b_cpu(Shape{{3, 2}}, TensorOptions{Dtype::Float32, Device::CPU});
    
    a_cpu.set_data(a_data);
    b_cpu.set_data(b_data);

    // 3. Display the input matrices
    std::cout << "\nMatrix A (2x3):" << std::endl;
    a_cpu.display(std::cout, 4);

    std::cout << "\nMatrix B (3x2):" << std::endl;
    b_cpu.display(std::cout, 4);

    // 4. Perform the CPU matrix multiplication
    std::cout << "\nPerforming matmul(A, B)..." << std::endl;
    a_cpu += b_cpu;

    // 5. Display the result
    // Expected result: [[58, 64], [139, 154]]
    std::cout << "\nResult C (2x2):" << std::endl;
    a_cpu.display(std::cout, 4);

    std::cout << "\n--- Test Complete ---" << std::endl;

    return 0;
}