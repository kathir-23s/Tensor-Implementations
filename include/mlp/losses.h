#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace mlp {

    /**
     * @brief Mean Squared Error (MSE) Loss
     * L = 1/N * sum((y - y_hat)^2)
     * 
     * @param predictions Predicted values
     * @param targets Target values
     * @return Tensor Scalar loss
     */
    Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

    /**
     * @brief Mean Absolute Error (MAE) Loss
     * L = 1/N * sum(|y - y_hat|)
     * 
     * @param predictions Predicted values
     * @param targets Target values
     * @return Tensor Scalar loss
     */
    Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

    /**
     * @brief Binary Cross Entropy (BCE) Loss
     * L = -1/N * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
     * 
     * @param predictions Predicted probabilities (0 < p < 1)
     * @param targets Binary targets (0 or 1)
     * @return Tensor Scalar loss
     */
    Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

    /**
     * @brief Categorical Cross Entropy (CCE) Loss
     * L = -1/N * sum(sum(y_c * log(y_hat_c)))
     * 
     * @param predictions Predicted probabilities (after Softmax)
     * @param targets One-hot encoded targets
     * @return Tensor Scalar loss
     */
    Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets);

}
}
