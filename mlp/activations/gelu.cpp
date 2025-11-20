#include "mlp/activations.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "dtype/Dtype.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor
{
    namespace mlp 
    {
        Tensor GeLU(const Tensor& input)
        {
            // GELU tanh approximation: 0.5 * x * [1 + tanh(√(2/π) * (x + 0.044715 * x³))]
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            Tensor half_x = 0.5f * input;
            Tensor x_cubed = input * input * input;
            Tensor tanh_input = sqrt_2_over_pi * (input + 0.044715f * x_cubed);
            Tensor inner_output = 1.0f + tanh(tanh_input);
            Tensor output = half_x * inner_output;
            return output;
        }
    }
}