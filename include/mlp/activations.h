#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace mlp {

    /**
     * @brief Applies the Softmax function to an n-dimensional input Tensor.
     * 
     * Rescales elements so that the elements of the n-dimensional output Tensor
     * lie in the range [0,1] and sum to 1.
     * 
     * @param input The input tensor.
     * @param dim The dimension along which Softmax will be computed (default: -1, last dimension).
     * @return Tensor The resulting tensor with the same shape as input.
     */
    Tensor softmax(const Tensor& input, int64_t dim = -1);

    /**
     * @brief Applies the Hyperbolic Tangent (Tanh) function element-wise.
     * 
     * @param input The input tensor.
     * @return Tensor The resulting tensor with the same shape as input.
     */
    Tensor tanh(const Tensor& input);  
    
    /**
     * @brief Applies the Sigmoid function element-wise.
     * 
     * @param input The input tensor.
     * @return Tensor The resulting tensor with the same shape as input.
     */
    Tensor sigmoid(const Tensor& input);

    /**
     * @brief Applies the Rectified Linear Unit (ReLU) function element-wise.
     * 
     * @param input The input tensor.
     * @return Tensor The resulting tensor with the same shape as input.
     */
    Tensor ReLU(const Tensor& input);


    /**
     * @brief Applies the Gaussian Error Linear Unit (GeLU) function element-wise.
     * 
     * @param input The input tensor.
     * @return Tensor The resulting tensor with the same shape as input.
     */
    Tensor GeLU(const Tensor& input);

    
} // namespace mlp
} // namespace OwnTensor
