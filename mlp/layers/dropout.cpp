#include "mlp/layers.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace mlp {

    Tensor dropout(const Tensor& input, float p) {
        if (p <= 0.0f || p >= 1.0f) {
             if (p >= 1.0f) return Tensor::zeros(input.shape(), TensorOptions().with_dtype(input.to_cpu().dtype())); 
             return input;
        }

        // 1. Create random mask [0, 1)
        Tensor mask = Tensor::rand(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()));

        // 2. Apply threshold: mask > p
        Tensor p_tensor = Tensor::full(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()), p);
        
        // Convert boolean to Int32 for where condition
        Tensor condition = (mask > p_tensor).as_type(Dtype::Int32);
        Tensor keep_mask = Tensor::where(condition, 1.0f, 0.0f);

        // 3. Scaling factor
        float scale_val = 1.0f / (1.0f - p);
        Tensor scale = Tensor::full(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()), scale_val);

        // 4. Apply
        return input * keep_mask * scale;
    }
}
}
