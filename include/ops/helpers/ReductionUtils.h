#pragma once

#include "core/Tensor.h" // Provides OwnTensor::Tensor and OwnTensor::Shape
#include <vector>
#include <cstdint>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::find, std::sort, etc.
#include <set>       // For unique axes check
#include <stdexcept> // For runtime_error

namespace OwnTensor {
namespace detail { // <<< START OF THE INTERNAL DETAIL NAMESPACE

// ========================================================================
// STACK-BASED HELPERS (INLINE - NO REDEFINITION ISSUES)
// ========================================================================

/**
 * @brief Stack-allocated unravel_index (no heap allocations)
 * INLINE means this is compiled into each translation unit that uses it.
 * NO redefinition errors because of inline keyword.
 */
inline void unravel_index_stack(int64_t linear_index,
                                const int64_t* shape_data,
                                size_t ndim,
                                int64_t* out_coords) {
    int64_t temp_index = linear_index;
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        if (shape_data[d] == 0) {
            out_coords[d] = 0;
            continue;
        }
        out_coords[d] = temp_index % shape_data[d];
        temp_index /= shape_data[d];
    }
}

/**
 * @brief Stack-based ravel_index (no allocations)
 * INLINE means this is compiled into each translation unit.
 */
inline int64_t ravel_index_stack(const int64_t* coords,
                                 const int64_t* strides,
                                 size_t ndim) {
    int64_t linear_index = 0;
    for (size_t i = 0; i < ndim; ++i) {
        linear_index += coords[i] * strides[i];
    }
    return linear_index;
}

// ========================================================================
// ORIGINAL FUNCTIONS (DECLARATIONS ONLY - IMPLEMENTED IN .CPP)
// ========================================================================

/**
 * @brief Normalizes the input axes to positive indices (0 to N-1) and validates them.
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 */
std::vector<int64_t> normalize_axes(const std::vector<int64_t>& input_dims, 
                                    const std::vector<int64_t>& axes);

/**
 * @brief Calculates the shape of the output tensor after reduction.
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 */
Shape calculate_output_shape(const std::vector<int64_t>& input_dims, 
                             const std::vector<int64_t>& normalized_axes, 
                             bool keepdim);

/**
 * @brief Calculates the total number of elements that will be combined for each reduction slice.
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 */
int64_t calculate_reduced_count(const std::vector<int64_t>& input_dims, 
                                const std::vector<int64_t>& normalized_axes);

/**
 * @brief Converts a linear index to a multi-dimensional coordinate vector (HEAP VERSION).
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 * NOTE: This version allocates a vector. Use unravel_index_stack for performance-critical paths.
 */
std::vector<int64_t> unravel_index(int64_t linear_index, 
                                   const std::vector<int64_t>& shape);

/**
 * @brief Converts a multi-dimensional coordinate vector back to a linear index using strides (HEAP VERSION).
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 * NOTE: This version takes vectors. Use ravel_index_stack for performance-critical paths.
 */
int64_t ravel_index(const std::vector<int64_t>& coords, 
                   const std::vector<int64_t>& strides);

} // namespace detail
} // namespace OwnTensor