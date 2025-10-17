#include <iostream>
#include <vector>
#include "../../include/Tensor.h"
#include "../../include/Types.h"
#include "../../include/TensorUnaryOps.hpp"   // your API header
#include "../../include/UnaryOps/exp_log.hpp"

using namespace OwnTensor;

int main() {
    // Assuming the Tensor constructor correctly initializes a single element '5'
    Tensor in {{{5}}, Dtype::Float64, Device::CPU}; 
    std::vector<double> data {0.0, 1.0, 2.0, 3.0, 4.0};
    //std::vector<int64_t>data{1,2,3,4,5};
    in.set_data(data);
    //Tensor out = exp(in); // out should contain e^5 (approx 148.413)
    exp_(in);
    in.display(std::cout, 2); 
    return 0;
}