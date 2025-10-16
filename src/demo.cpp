#include <iostream>
#include "../include/Tensor.h"

using namespace OwnTensor;
int main() {
    //Tensor t({{5}},OwnTensor::Dtype::Float32,OwnTensor::DeviceIndex(OwnTensor::Device::CUDA));
    Tensor gpu_tensor(Shape{{2}}, Dtype::Float32, DeviceIndex(Device::CUDA), false);
    std::vector<float>data{0.0f,1.0f};
    gpu_tensor.set_data(data);
    gpu_tensor.display(std::cout,0);
    return 0;
}