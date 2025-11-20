#include "mlp/losses.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Arithmetics.h" // For abs
#include "ops/UnaryOps/Reduction.h"   // For reduce_mean

namespace OwnTensor {
namespace mlp {

    Tensor mae_loss(const Tensor& predictions, const Tensor& targets) {
        // L = mean(|y - y_hat|)
        Tensor diff = predictions - targets;
        Tensor abs_diff = OwnTensor::abs(diff, 0); // stream 0
        return OwnTensor::reduce_mean(abs_diff);
    }

}
}
