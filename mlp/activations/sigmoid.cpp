#include "mlp/activations.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "dtype/Dtype.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor {
namespace mlp {

    Tensor sigmoid(const Tensor& input) {
        
        // 1. Calculating exponent
        Tensor exp_input = exp(input);
        #ifdef WITH_DISPLAY
            exp_input.display();
        #endif
        // 2. Denominator
        Tensor denom = 1 + exp_input;
        #ifdef WITH_DISPLAY
            denom.display();
        #endif
        // 3. Sigmoid output
        Tensor output = exp_input / denom;
        #ifdef WITH_DISPLAY
            output.display();
        #endif
        return output;
    }


} // namespace mlp
} // namespace OwnTensor