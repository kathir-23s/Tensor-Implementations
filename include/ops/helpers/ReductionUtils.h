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

// NOTE: All functions now implicitly return/accept OwnTensor::Shape

/**
 * @brief Normalizes the input axes to positive indices (0 to N-1) and handles empty axes (full reduction).
 * @param input_dims The shape/dimensions of the input tensor.
 * @param axes The dimensions to reduce over (can include negative indices).
 * @return A vector of positive, unique axis indices.
 */
std::vector<int64_t> normalize_axes(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& axes);

/**
 * @brief Calculates the shape of the output tensor after reduction.
 * @param input_dims The shape of the input tensor.
 * @param normalized_axes The axes being reduced.
 * @param keepdim If true, keeps reduced dimensions as 1.
 * @return The Shape struct of the output tensor.
 */
Shape calculate_output_shape(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes, bool keepdim);

/**
 * @brief Calculates the total number of elements that will be combined for each reduction slice.
 * @param input_dims The shape of the input tensor dimensions.
 * @param normalized_axes The axes being reduced.
 * @return The total number of elements being reduced.
 */
int64_t calculate_reduced_count(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes);

/**
 * @brief Converts a linear index to a multi-dimensional coordinate vector.
 * @param linear_index The 1D index.
 * @param shape The shape of the tensor.
 * @return A vector of coordinates (e.g., {i, j, k}).
 */
std::vector<int64_t> unravel_index(int64_t linear_index, const std::vector<int64_t>& shape);

/**
 * @brief Converts a multi-dimensional coordinate vector back to a linear index using strides.
 * @param coords The multi-dimensional coordinates.
 * @param strides The strides of the tensor's shape.
 * @return The 1D linear index.
 */
int64_t ravel_index(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides);


} // namespace detail
} // namespace OwnTensor
