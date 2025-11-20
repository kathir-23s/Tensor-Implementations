#include "mlp/layers.h"
#include <vector>
#include <numeric>

namespace OwnTensor {
namespace mlp {

    Tensor flatten(const Tensor& input) {
        // Reshape (N, d1, d2, ...) -> (N, D)
        
        const auto& dims = input.shape().dims;
        if (dims.empty()) {
            return input; // Scalar or empty
        }

        int64_t batch_size = dims[0];
        int64_t total_features = 1;

        // Calculate product of all other dimensions
        for (size_t i = 1; i < dims.size(); ++i) {
            total_features *= dims[i];
        }

        return input.reshape({{batch_size, total_features}});
    }

}
}
