#include "TensorLib.h"

using namespace OwnTensor;

auto& print = std::cout;

int main()
{
    Shape shape = {{2,2}};
    TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = Device::CPU;
    opts.requires_grad = false;
    
    Tensor predictions = Tensor::rand(shape, opts);
    predictions.display();
    print << "\n";

    Tensor targets(shape, opts);
    targets.set_data({-1.0f, 1.0f, 0.f, 0.5f});
    targets.display();
    print << "\n";

    {
    print << "===================================" << std::endl;
    print << " Testing BINARY CROSS ENTROPY LOSS " << std::endl;
    print << "===================================" << std::endl;

    Tensor result = mlp::binary_cross_entropy(predictions, targets);
    result.display();
    }
    print << "\n";
    {
    print << "=================================" << std::endl;
    print << " Testing MEAN SQUARED ERROR LOSS " << std::endl;
    print << "=================================" << std::endl;

    Tensor result = mlp::mse_loss(predictions, targets);
    result.display();
    }
    print << "\n";
    {
    print << "===================================" << std::endl;
    print << " Testing CATEGORICAL CROSS ENTROPY " << std::endl;
    print << "===================================" << std::endl;

    Tensor result = mlp::categorical_cross_entropy(predictions, targets);
    result.display();
    }
    print << "\n";
    {
    print << "==================================" << std::endl;
    print << " Testing MEAN ABSOLUTE ERROR LOSS " << std::endl;
    print << "==================================" << std::endl;

    Tensor result = mlp::mae_loss(predictions, targets);
    result.display();
    }
}