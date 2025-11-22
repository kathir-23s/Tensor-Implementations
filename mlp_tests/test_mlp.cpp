#include "core/Tensor.h"
#include "mlp/layers.h"
#include "mlp/losses.h"
#include "mlp/activations.h" // For ReLU, Softmax
#include <iostream>
#include <vector>

using namespace OwnTensor;

int main() {
    // 1. Setup Data
    std::cout << "Initializing Data..." << std::endl;
    int64_t batch_size = 32;
    int64_t input_features = 784;
    int64_t hidden_units = 128;
    int64_t output_classes = 10;

    TensorOptions options;
    options.device = Device::CUDA;
    options.dtype = Dtype::Float32;

    // Input: (Batch, Features)
    Tensor input = Tensor::randn({{batch_size, input_features}}, options);
    std::cout << "input created" << std::endl;
    
    // Weights and Biases for Layer 1
    Tensor w1 = Tensor::randn({{hidden_units, input_features}}, options);
    std::cout << "w1 created" << std::endl;

    Tensor b1 = Tensor::zeros({{hidden_units}}, options);
    std::cout << "b1 created" << std::endl;

    // Weights and Biases for Layer 2
    Tensor w2 = Tensor::randn({{output_classes, hidden_units}}, options);
    std::cout << "w2 created" << std::endl;

    Tensor b2 = Tensor::zeros({{output_classes}}, options);
    std::cout << "b2 created" << std::endl;

    // Targets: One-hot encoded (Batch, Classes)
    Tensor targets = Tensor::randn({{batch_size, output_classes}}, options);
    std::cout << "targets created" << std::endl;


    // Dummy targets from rand function and normalizing it
    targets = mlp::softmax(targets, 1);
    std::cout << "normalized targets created" << std::endl;


    // 2. Forward Pass
    std::cout << "Running Forward Pass..." << std::endl;
    
    // Layer 1: Linear -> ReLU
    Tensor x = mlp::linear(input, w1, b1);
    x = mlp::ReLU(x);
    
    // Dropout (Training mode)
    x = mlp::dropout(x, 0.5f);

    // Layer 2: Linear -> Softmax
    Tensor logits = mlp::linear(x, w2, b2);
    Tensor output = mlp::softmax(logits, 1);

    std::cout << "Output Shape: [";
    for (auto d : output.shape().dims) std::cout << d << " ";
    std::cout << "]" << std::endl;

    // 3. Loss Calculation
    std::cout << "Calculating Loss..." << std::endl;
    
    Tensor loss = mlp::categorical_cross_entropy(output, targets);
    
    std::cout << "Loss Value: ";
    loss.display(); 

    // Check other losses just to verify they run
    Tensor mse = mlp::mse_loss(output, targets);
    std::cout << "MSE Loss: ";
    mse.display();

    std::cout << "Test Complete." << std::endl;
    return 0;
}
