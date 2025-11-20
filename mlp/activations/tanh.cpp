#include "mlp/activations.h"
#include "ops/UnaryOps/Trigonometry.h"


namespace OwnTensor {
namespace mlp {

    Tensor tanh(const Tensor& input) {
        Tensor output = OwnTensor::Trig::tanh(input);
        return output;
    }



} // namespace mlp
} // namespace OwnTensor
