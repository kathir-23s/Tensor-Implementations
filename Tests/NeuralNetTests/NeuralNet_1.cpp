#include "TensorLib.h"
#include <iostream>

using namespace OwnTensor;


void test_mlp_forward_pass() {
    std::cout << "=== Basic MLP Forward Pass Test ===\n";
    
    // Create a simple network: 3 inputs -> 4 hidden -> 2 outputs
    int input_samples = 10;
    int input_size = 3;
    int hidden_size = 4;
    int output_size = 2;
    
    TensorOptions opts;
    opts.dtype = Dtype::Int32;
    opts.device = DeviceIndex(Device::CPU);
    opts.requires_grad = false;

    // Input data
    Tensor input = Tensor::full({{input_samples, input_size}}, opts, 14);
    input.display(std::cout, 12);
    std::cout << "input tensor created successfully\n" << std::endl;
    
    // Layer 1 weights and bias
    Tensor W1= Tensor::full({{input_size, hidden_size}}, opts, 21);
    std::cout << "Layer 1 weight tensor created successfully\n" << std::endl;
    W1.display(std::cout, 12);

    Tensor b1 = Tensor::ones({{1, hidden_size}}, opts);

    b1.display(std::cout, 12);
    std::cout << "Layer 1 bias tensor created successfully\n" << std::endl;

    // Layer 2 weights and bias  
    Tensor W2= Tensor::full({{hidden_size, output_size}}, opts, 3);

    W2.display(std::cout, 12);
    std::cout << "Layer 2 weight tensor created successfully\n" << std::endl;
    
    Tensor b2 = Tensor::ones({{1, output_size}}, opts);

    b2.display(std::cout, 12);
    std::cout << "Layer 2 bias tensor created successfully\n" << std::endl;
    
    // Forward pass
    // Layer 1: input @ W1 + b1
    Tensor h1 = matmul(input, W1) + b1; // + b1; 
    std::cout << "WX + b done successfully\n" << std::endl;
    // h1 += b1;  // Using your tensor ops
    h1.display(std::cout, 12);

    
    // Activation (tanh)
    Tensor a1 = tanh(h1);
    // a1.display(std::cout, 12);

    // Layer 2: a1 @ W2 + b2  
    Tensor output = matmul(h1, W2) + b2;
    
    // Display results
    std::cout << "Input shape: ";
    for (auto dim : input.shape().dims) std::cout << dim << " ";
    std::cout << "\n";
    
    std::cout << "Output shape: ";
    for (auto dim : output.shape().dims) std::cout << dim << " ";
    std::cout << "\n";
    
    std::cout << "Output values:\n";
    // Tensor output_to_cpu = output.to_cpu();
    output.display(std::cout, 12);
    
    std::cout << "\nTest completed successfully!\n\n";
}

int main() 
{
    test_mlp_forward_pass();
    return 0;
}