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
    std::cin.get(); // 1
    // 2. Construct GPU tensor (Float32, CUDA device)
    Tensor input({{5}}, Dtype::Float32, Device::CUDA);
    std::cin.get(); // 2
    input.set_data(host_data);  // Should handle host-to-device copy internally
    std::cin.get(); // 3
    // 3. Call the high-level API
    Tensor output = exp(input); // this should dispatch to exp_gpu_wrap()
    std::cin.get(); // 4
    // 4. Copy result back to CPU for checking
    Tensor result = output.to_cpu(); // hypothetical utility (you likely already have something like this)
    std::cin.get(); // 5
    // 5. Display results
    // std::cout << "Input : ";
    // input.display(std::cout, 5);
    std::cout << "\nOutput (expf): ";
    result.display(std::cout, 5);
    std::cout << std::endl;
    // Verify data matches
    // const float* retrieved_data = static_cast<const float*>(cpu_tensor.data());
    // bool data_matches = true;
    // for (size_t i = 0; i < source_data.size(); ++i) {
    //     if (retrieved_data[i] != source_data[i]) {
    //         data_matches = false;
    //         cout << "Mismatch at index " << i << ": expected " << source_data[i] 
    //                 << ", got " << retrieved_data[i] << endl;
    //         break;
    //     }
    // }
    
    // assert(data_matches);
    // cout << "✓ Float data transfer successful" << endl;

    return 0;
}