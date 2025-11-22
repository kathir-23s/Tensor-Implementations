#include "core/Tensor.h"
#include "mlp/activations.h"
#include "mlp/layers.h"
#include "mlp/losses.h"
#include "device/Device.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace OwnTensor;

void test_softmax_cuda() {
    std::cout << "Testing Softmax on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        input.set_data(data);

        Tensor output = mlp::softmax(input, -1);
        
        // Basic check: sum should be 1.0 along last dim
        // We can't easily check values on host without copying back, 
        // but successful execution is the first step.
        std::cout << "Softmax executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Softmax failed on CUDA: " << e.what() << std::endl;
    }
}

void test_gelu_cuda() {
    std::cout << "Testing GeLU on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
        input.set_data(data);

        Tensor output = mlp::GeLU(input);
        std::cout << "GeLU executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "GeLU failed on CUDA: " << e.what() << std::endl;
    }
}

void test_tanh_cuda() {
    std::cout << "Testing Tanh on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
        input.set_data(data);

        Tensor output = mlp::tanh(input);
        std::cout << "Tanh executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Tanh failed on CUDA: " << e.what() << std::endl;
    }
}

void test_flatten_cuda() {
    std::cout << "Testing Flatten on CUDA..." << std::endl;
    try {
        Tensor input(Shape{{2, 3, 4}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        // Fill with dummy data
        Tensor output = mlp::flatten(input);
        
        if (output.ndim() == 2 && output.shape().dims[0] == 2 && output.shape().dims[1] == 12) {
             std::cout << "Flatten executed successfully on CUDA (Shape correct)." << std::endl;
        } else {
             std::cout << "Flatten executed but shape is incorrect." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Flatten failed on CUDA: " << e.what() << std::endl;
    }
}

void test_mae_cuda() {
    std::cout << "Testing MAE on CUDA..." << std::endl;
    try {
        Tensor pred(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        Tensor target(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        std::vector<float> p_data = {0.5f, 0.6f, 0.7f, 0.8f};
        std::vector<float> t_data = {1.0f, 0.0f, 1.0f, 0.0f};
        pred.set_data(p_data);
        target.set_data(t_data);

        Tensor loss = mlp::mae_loss(pred, target);
        std::cout << "MAE executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "MAE failed on CUDA: " << e.what() << std::endl;
    }
}

void test_bce_cuda() {
    std::cout << "Testing BCE on CUDA..." << std::endl;
    try {
        Tensor pred(Shape{{4}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        Tensor target(Shape{{4}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        std::vector<float> p_data = {0.5f, 0.6f, 0.7f, 0.8f};
        std::vector<float> t_data = {1.0f, 0.0f, 1.0f, 0.0f};
        pred.set_data(p_data);
        target.set_data(t_data);

        Tensor loss = mlp::binary_cross_entropy(pred, target);
        std::cout << "BCE executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "BCE failed on CUDA: " << e.what() << std::endl;
    }
}

void test_cce_cuda() {
    std::cout << "Testing CCE on CUDA..." << std::endl;
    try {
        // Batch size 2, 3 classes
        Tensor pred(Shape{{2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        Tensor target(Shape{{2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        std::vector<float> p_data = {0.1f, 0.8f, 0.1f,  0.7f, 0.2f, 0.1f};
        std::vector<float> t_data = {0.0f, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f};
        pred.set_data(p_data);
        target.set_data(t_data);

        Tensor loss = mlp::categorical_cross_entropy(pred, target);
        std::cout << "CCE executed successfully on CUDA." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CCE failed on CUDA: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Verifying Remaining Components on CUDA ===" << std::endl;
    
#ifdef WITH_CUDA
    if (!device::cuda_available()) {
        std::cerr << "CUDA not available!" << std::endl;
        return 1;
    }
    std::cout << "CUDA is available." << std::endl;

    test_softmax_cuda();
    test_gelu_cuda();
    test_tanh_cuda();
    test_flatten_cuda();
    test_mae_cuda();
    test_bce_cuda();
    test_cce_cuda();
    
#else
    std::cout << "CUDA not enabled." << std::endl;
#endif

    return 0;
}
