#include "ops/helpers/ReductionUtils.h" // Provides declarations for all functions
#include "core/Tensor.h"          // Provides OwnTensor::Shape (required for calculate_output_shape return type)
#include <algorithm> //For std::find
#include <numeric>   // For std::accumulate, std::multiplies
#include <set>       // For unique axes
#include <stdexcept> // For runtime_error
#include <cstdint>   // For int64_t and size_t

namespace OwnTensor {
namespace detail {

/**
 * @brief Normalizes the input axes to positive indices (0 to N-1) and validates them.
 * @param input_dims The shape/dimensions of the input tensor.
 * @param axes The dimensions to reduce over (can include negative indices).
 * @return A vector of positive, unique axis indices, sorted ascendingly.
 */
std::vector<int64_t> normalize_axes(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& axes) {
    const int64_t ndim = input_dims.size();
    std::set<int64_t> unique_axes_set;

    // Handle empty axes: means reduce over all dimensions
    if (axes.empty()) {
        for (int64_t i = 0; i < ndim; ++i) {
            unique_axes_set.insert(i);
        }
    } else {
        for (int64_t axis : axes) {
            int64_t normalized_axis = axis;
            
            // 1. Handle negative axes (e.g., -1 becomes ndim - 1)
            if (normalized_axis < 0) {
                normalized_axis += ndim;
            }

            // 2. Validate bounds
            if (normalized_axis < 0 || normalized_axis >= ndim) {
                throw std::runtime_error("Reduction axis " + std::to_string(axis) + 
                                         " is out of bounds for tensor of rank " + std::to_string(ndim) + ".");
            }

            unique_axes_set.insert(normalized_axis);
        }
    }

    // Convert the set back to a sorted vector
    std::vector<int64_t> normalized_axes(unique_axes_set.begin(), unique_axes_set.end());
    return normalized_axes;
}

/**
 * @brief Calculates the shape of the output tensor after reduction.
 * @param input_dims The shape of the input tensor.
 * @param normalized_axes The axes being reduced (positive indices).
 * @param keepdim If true, keeps reduced dimensions as 1.
 * @return The Shape struct of the output tensor.
 */
Shape calculate_output_shape(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes, bool keepdim) {
    std::vector<int64_t> output_dims;
    
    // Lambda to check if a dimension index is marked for reduction
    auto is_reduced = [&](int64_t dim_idx) {
        return std::find(normalized_axes.begin(), normalized_axes.end(), dim_idx) != normalized_axes.end();
    };

    // Use size_t for the loop counter 'i' to avoid signed/unsigned comparison warnings.
    for (size_t i = 0; i < input_dims.size(); ++i) {
        if (is_reduced(static_cast<int64_t>(i))) {
            if (keepdim) {
                // Reduced axis is kept as size 1
                output_dims.push_back(1);
            }
            // If keepdim is false, this axis is dropped
        } else {
            // Unreduced axis, keep original size
            output_dims.push_back(input_dims[i]);
        }
    }
    
    // Handle scalar output 
    if (output_dims.empty()) {
        output_dims.push_back(1);
    }

    return Shape{output_dims};
}

/**
 * @brief Calculates the total number of elements that will be combined for each reduction slice.
 * @param input_dims The shape of the input tensor dimensions.
 * @param normalized_axes The axes being reduced.
 * @return The total number of elements being reduced.
 */
int64_t calculate_reduced_count(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes) {
    // If normalized_axes is empty, it means reduce over all dimensions
    if (normalized_axes.empty()) {
        return std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());
    }
    
    int64_t count = 1;
    for (int64_t axis : normalized_axes) {
        count *= input_dims[axis];
    }
    return count;
}


/**
 * @brief Converts a linear index to a multi-dimensional coordinate vector (Unravels).
 * This assumes C-order (row-major) layout.
 * @param linear_index The 1D index.
 * @param shape The shape of the tensor.
 * @return A vector of coordinates (e.g., {i, j, k}).
 */
std::vector<int64_t> unravel_index(int64_t linear_index, const std::vector<int64_t>& shape) {
    std::vector<int64_t> coords(shape.size());
    int64_t temp_index = linear_index;

    // Iterate backwards for C-order (row-major), using size_t for the loop index.
    for (size_t i = shape.size(); i-- > 0; ) {
        if (shape[i] == 0) continue; 
        coords[i] = temp_index % shape[i];
        temp_index /= shape[i];
    }
    
    return coords;
}

/**
 * @brief Converts a multi-dimensional coordinate vector back to a linear index using strides (Ravel).
 * @param coords The multi-dimensional coordinates.
 * @param strides The strides of the tensor's shape.
 * @return The 1D linear index.
 */
int64_t ravel_index(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides) {
    if (coords.size() != strides.size()) {
        throw std::runtime_error("Coordinate vector and stride vector must have the same rank.");
    }
    
    int64_t linear_index = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        linear_index += coords[i] * strides[i];
    }
    return linear_index;
}

} // namespace detail
} // namespace OwnTensor