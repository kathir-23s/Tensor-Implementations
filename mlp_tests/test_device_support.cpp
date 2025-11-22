#include "core/Tensor.h"
#include "mlp/activations.h"
#include "mlp/layers.h"
#include "mlp/losses.h"
#include "ops/ScalarOps.h"
#include "device/Device.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace OwnTensor;

void test_relu_cuda() {
    std::cout << "Testing ReLU on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
        input.set_data(data);

        Tensor output = mlp::ReLU(input);
        
        std::cout << "ReLU executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ReLU failed on CUDA: " << e.what() << std::endl;
    }
}

void test_linear_cuda() {
    std::cout << "Testing Linear on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        input.fill(1.0f);
        
        Tensor weights(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        weights.fill(0.5f);
        
        Tensor bias(Shape{{2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        bias.fill(0.1f);

        Tensor output = mlp::linear(input, weights, bias);
        std::cout << "Linear executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Linear failed on CUDA: " << e.what() << std::endl;
    }
}

void test_mse_cuda() {
    std::cout << "Testing MSE on CUDA..." << std::endl;
    try {
        Tensor pred(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        pred.fill(1.0f);
        
        Tensor target(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        target.fill(0.5f);

        Tensor loss = mlp::mse_loss(pred, target);
        std::cout << "MSE executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "MSE failed on CUDA: " << e.what() << std::endl;
    }
}

void test_sigmoid_cuda() {
    std::cout << "Testing Sigmoid on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
        input.set_data(data);

        Tensor output = mlp::sigmoid(input);
        std::cout << "Sigmoid executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Sigmoid failed on CUDA: " << e.what() << std::endl;
    }
}

void test_dropout_cuda() {
    std::cout << "Testing Dropout on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{10, 10}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        input.fill(1.0f);

        // Dropout with p=0.5
        Tensor output = mlp::dropout(input, 0.5f);
        std::cout << "Dropout executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Dropout failed on CUDA: " << e.what() << std::endl;
    }
}

int main() {
#ifdef WITH_CUDA
    test_relu_cuda();
    test_linear_cuda();
    test_mse_cuda();
    test_sigmoid_cuda();
    test_dropout_cuda();
#else
    std::cout << "CUDA not enabled." << std::endl;
#endif
    return 0;
}
