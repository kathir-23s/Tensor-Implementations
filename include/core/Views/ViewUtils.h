#pragma once

#include "core/Tensor.h"  // This is the correct path

namespace OwnTensor {
namespace ViewUtils {

    // Computes row-major strides from a shape
    Stride compute_strides(const Shape& shape);
    
    // Checks if new shape has same number of elements
    bool is_shape_compatible(size_t numel, const Shape& new_shape);
    
    // Replaces -1 dimension with computed value
    void infer_dimension(size_t numel, Shape& shape);

    // Dimension helpers
    int normalize_dim(int dim, int ndim);                 // negative -> positive, bounds check throws
    void swap_dimensions(Shape& shape, Stride& stride, int dim0, int dim1);

    // Flatten helpers
    int64_t compute_flatten_size(const Shape& shape, int start_dim, int end_dim);
    Shape compute_flatten_shape(const Shape& shape, int start_dim, int end_dim);

    // Unflatten helpers
    void validate_unflatten(const Shape& shape, int dim, const Shape& sizes); // throws on mismatch
    Shape compute_unflatten_shape(const Shape& shape, int dim, const Shape& sizes);
    Stride compute_unflatten_strides(const Stride& old_stride, int dim, const Shape& sizes);
} // namespace ViewUtils
} // namespace OwnTensor
