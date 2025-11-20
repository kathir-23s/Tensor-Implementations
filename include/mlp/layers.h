#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace mlp {

    /**
     * @brief Linear (Dense) Layer
     * Performs Y = X * W^T + b
     * 
     * @param input Input tensor of shape (batch_size, in_features)
     * @param weights Weight tensor of shape (out_features, in_features)
     * @param bias Bias tensor of shape (out_features) or (1, out_features)
     * @return Tensor Output tensor of shape (batch_size, out_features)
     */
    Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias);

    /**
     * @brief Flatten Layer
     * Reshapes multi-dimensional inputs into a flat vector (keeping batch dimension).
     * 
     * @param input Input tensor of shape (N, d1, d2, ...)
     * @return Tensor Output tensor of shape (N, D) where D = d1 * d2 * ...
     */
    Tensor flatten(const Tensor& input);

    /**
     * @brief Dropout Layer
     * Randomly zeroes some elements of the input tensor with probability p using inverted dropout.
     * 
     * @param input Input tensor
     * @param p Probability of an element to be zeroed.
     * @return Tensor Output tensor
     */
    Tensor dropout(const Tensor& input, float p = 0.5f);

}
}
