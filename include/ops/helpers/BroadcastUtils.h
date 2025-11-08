#pragma once
#include "core/Tensor.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace OwnTensor {

inline std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& shape_a, 
                                          const std::vector<int64_t>& shape_b) {
    size_t ndim_a = shape_a.size();
    size_t ndim_b = shape_b.size();
    size_t max_ndim = std::max(ndim_a, ndim_b);
    
    std::vector<int64_t> result(max_ndim);
    
    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = (i < ndim_a) ? shape_a[ndim_a - 1 - i] : 1;
        int64_t dim_b = (i < ndim_b) ? shape_b[ndim_b - 1 - i] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        
        result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
    }
    
    return result;
}

inline bool broadcast_compatible(const std::vector<int64_t>& shape_a,
                               const std::vector<int64_t>& shape_b) {
    try {
        broadcast_shape(shape_a, shape_b);
        return true;
    } catch (...) {
        return false;
    }
}

}