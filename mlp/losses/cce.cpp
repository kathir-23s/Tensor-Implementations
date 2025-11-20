#include "mlp/losses.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Exponents.h"   // For log
#include "ops/UnaryOps/Reduction.h"   // For reduce_sum, reduce_mean
#include <vector>

namespace OwnTensor {
namespace mlp {

    Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets) {
        float epsilon_val = 1e-7f;
        Tensor epsilon = Tensor::full(predictions.shape(), TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
        Tensor one_minus_epsilon = Tensor::full(predictions.shape(), TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);
        
        // Clip predictions (convert bool conditions to Int32)
        Tensor clipped_preds = Tensor::where((predictions < epsilon).as_type(Dtype::Int32), epsilon, predictions);
        clipped_preds = Tensor::where((clipped_preds > one_minus_epsilon).as_type(Dtype::Int32), one_minus_epsilon, clipped_preds);

        Tensor log_preds = OwnTensor::log(clipped_preds);
        
        Tensor target_log_probs = targets * log_preds;
        
        std::vector<int64_t> axis = {1};
        Tensor sample_losses = OwnTensor::reduce_sum(target_log_probs, axis);
        
        Tensor neg_one = Tensor::full({{1}}, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
        return OwnTensor::reduce_mean(sample_losses) * neg_one;
    }

}
}
