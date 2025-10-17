#include <iostream>
#include <vector>
#include <cassert>
#include "../../include/Tensor.h"
#include "../../include/Types.h"
#include "../../include/TensorUnaryOps.hpp"  // your high-level exp() API

using namespace OwnTensor;

int main() {
    // 1. Create a small input tensor on the GPU
    std::vector<float> host_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}; // currently in CPU
    
    // 2. Construct GPU tensor (Float32, CUDA device)
    Tensor input({{5}}, Dtype::Float32, Device::CUDA);
    std::cin.get();
    input.set_data(host_data);  // Should handle host-to-device copy internally
    
    // 3. Call the high-level API
    Tensor output = exp(input); // this should dispatch to exp_gpu_wrap()
    
    // 4. Copy result back to CPU for checking
    Tensor result = output.to_cpu(); // hypothetical utility (you likely already have something like this)
    
    std::cout << "\nOutput (expf): ";
    result.display(std::cout, 5);
    std::cout << std::endl;

    return 0;
}