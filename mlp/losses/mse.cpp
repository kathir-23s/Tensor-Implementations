#include "mlp/losses.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Arithmetics.h" // For pow
#include "ops/UnaryOps/Reduction.h"   // For reduce_mean

namespace OwnTensor {
namespace mlp {

    Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
        // L = mean((y - y_hat)^2)
        Tensor diff = predictions - targets;
        Tensor sq_diff = OwnTensor::pow(diff, 2, 0); // stream 0
        return OwnTensor::reduce_mean(sq_diff);
    }

}
}
