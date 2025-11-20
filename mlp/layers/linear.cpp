#include "mlp/layers.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h" // For matmul

namespace OwnTensor {
namespace mlp {

    Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias) {
        // Y = X * W^T + b
        
        // 1. Transpose weights: (out_features, in_features) -> (in_features, out_features)
        Tensor weights_t = weights.t();

        // 2. Matmul: (batch, in) * (in, out) -> (batch, out)
        Tensor output = OwnTensor::matmul(input, weights_t);

        // 3. Add bias: (batch, out) + (out) -> (batch, out) (Broadcasting)
        output = output + bias;

        return output;
    }

}
}
