#include "mlp/activations.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "dtype/Dtype.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor {
namespace mlp {

    Tensor ReLU(const Tensor& input) {
        Tensor condition = (input > 0.0f).as_type(Dtype::Int32);
        return Tensor::where(condition, input, 0.0f);
    }


} // namespace mlp
} // namespace OwnTensor
