#include <iostream>
#include <vector>
#include "../../include/Tensor.h"
#include "../../include/Types.h"
#include "../../include/TensorUnaryOps.hpp"   // your API header
#include "../../include/UnaryOps/exp_log.hpp"

using namespace OwnTensor;

int main() {
    // Assuming the Tensor constructor correctly initializes a single element '5'
    Tensor in {{{5}}, Dtype::Int32, Device::CPU}; 
    // std::vector<float16_t> data {float16_t(0.0f), float16_t(1.0f), float16_t(2.0f), float16_t(3.0f), float16_t(4.0f)};
    std::vector<int32_t>data{1,2,3,4,5};
    in.set_data(data);
    Tensor out = exp(in); // out should contain e^5 (approx 148.413)
    out.display(std::cout, 2); 
    return 0;
}