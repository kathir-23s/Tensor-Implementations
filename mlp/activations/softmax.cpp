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

    Tensor softmax(const Tensor& input, int64_t dim) {
        if (!input.is_valid()) {
            throw std::runtime_error("Input tensor is invalid (empty or uninitialized).");
        }

        int64_t ndim = input.ndim();
        if (ndim == 0) {
             throw std::runtime_error("Softmax cannot be applied to a scalar (0-d tensor).");
        }

        // Handle negative dimension index
        if (dim < 0) {
            dim += ndim;
        }

        if (dim < 0 || dim >= ndim) {
            throw std::out_of_range("Dimension out of range for softmax.");
        }

        // Numerical Stability: subtract max value before exp
        // x_safe = x - max(x)
        // softmax(x) = exp(x_safe) / sum(exp(x_safe))
        
        // 1. Find max along the dimension (keepdim=true to allow broadcasting)
        Tensor max_val = reduce_max(input, {dim}, true);
        #ifdef WITH_DISPLAY
            max_val.display();
        #endif

        // 2. Subtract max from input
        Tensor shifted_input = input - max_val;
        #ifdef WITH_DISPLAY
            shifted_input.display();
        #endif

        // 3. Compute exp
        Tensor exp_input = exp(shifted_input);
        #ifdef WITH_DISPLAY
            exp_input.display();
        #endif

        // 4. Compute sum of exp (keepdim=true to allow broadcasting)
        Tensor sum_exp = reduce_sum(exp_input, {dim}, true);
        #ifdef WITH_DISPLAY
            sum_exp.display();
        #endif

        // 5. Divide exp by sum
        Tensor output = exp_input / sum_exp;
        #ifdef WITH_DISPLAY
            output.display();
        #endif

        return output;
    }


} // namespace mlp
} // namespace OwnTensor
