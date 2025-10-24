#include "core/Views/ViewUtils.h"
#include <stdexcept>

namespace OwnTensor {
namespace ViewUtils {

// Computes row-major strides for a given shape
Stride compute_strides(const Shape& shape) {
    Stride result;
    // Empty shape = empty strides
    if (shape.dims.empty()) return result;
    // Allocate space for strides
    result.strides.resize(shape.dims.size());
    // Start with stride of 1 for the last dimension
    int64_t stride = 1;
    // Work backwards through dimensions
    for (int i = shape.dims.size() - 1; i >= 0; --i) {
        result.strides[i] = stride;
        stride *= shape.dims[i];
    }
    return result;
}

// Checks if new shape has same number of elements as original
bool is_shape_compatible(size_t numel, const Shape& new_shape){
    size_t new_numel = 1;
    int infer_count = 0;
    // Count total elements and number of -1s
    for (int64_t dim : new_shape.dims) {
        if (dim == -1) {
            infer_count++;
            continue;
        }
        new_numel *= dim;
    }
    // Can't have more than one -1
    if (infer_count > 1) return false;
    // If no -1, sizes must match exactly
    if (infer_count == 0) return numel == new_numel;
    // If one -1, check if numel is divisible by known dimensions
    return (numel % new_numel) == 0;
}

// Replaces -1 dimension with computed value
void infer_dimension(size_t numel, Shape& shape){
    int infer_idx = -1;
    size_t known_size = 1;
    // Find the -1 dimension and compute product of known dimensions
    for (size_t i = 0; i < shape.dims.size(); i++) {
        if (shape.dims[i] == -1) {
            if (infer_idx != -1) {
                throw std::runtime_error("Only one dimension can be inferred (-1)");
            }
            infer_idx = i;
        } else {
            known_size *= shape.dims[i];
        }
    }
    // If we found a -1, compute what it should be
    if (infer_idx != -1) {
        if (numel % known_size != 0) {
            throw std::runtime_error("Shape is not compatible with tensor size");
        }
        shape.dims[infer_idx] = numel / known_size;
    }
}


int normalize_dim(int dim, int ndim)
{
    int d = dim;
    if (d < 0) d += ndim;
    if (d < 0 || d >= ndim)
        throw std::runtime_error("Dimension out of range: " + std::to_string(dim) +
                                 " for ndim=" + std::to_string(ndim));
    return d;
}


void swap_dimensions(Shape& shape, Stride& stride, int dim0, int dim1)
{
    if (dim0 == dim1) return;
    std::swap(shape.dims[dim0], shape.dims[dim1]);
    std::swap(stride.strides[dim0], stride.strides[dim1]);
}


int64_t compute_flatten_size(const Shape& shape, int start_dim, int end_dim)
{
    if (shape.dims.empty())
        throw std::runtime_error("flatten: shape must have at least 1 dim");
    if (start_dim < 0 || end_dim < 0 || start_dim >= (int)shape.dims.size() || end_dim >= (int)shape.dims.size())
        throw std::runtime_error("flatten: start/end out of bounds");
    if (start_dim > end_dim)
        throw std::runtime_error("flatten: start_dim must be <= end_dim");

    int64_t prod = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
        if (shape.dims[i] <= 0)
            throw std::runtime_error("flatten: dimensions must be positive");
        prod *= shape.dims[i];
    }
    return prod;
}


Shape compute_flatten_shape(const Shape& shape, int start_dim, int end_dim)
{
    int ndim = static_cast<int>(shape.dims.size());
    start_dim = normalize_dim(start_dim, ndim);
    end_dim   = normalize_dim(end_dim,   ndim);
    if (start_dim > end_dim)
        throw std::runtime_error("flatten: start_dim must be <= end_dim");

    int64_t flat = compute_flatten_size(shape, start_dim, end_dim);

    Shape out;
    out.dims.reserve(ndim - (end_dim - start_dim));
    for (int i = 0; i < start_dim; ++i) out.dims.push_back(shape.dims[i]);
    out.dims.push_back(flat);
    for (int i = end_dim + 1; i < ndim; ++i) out.dims.push_back(shape.dims[i]);
    return out;
}


static int64_t product(const std::vector<int64_t>& v)
{
    int64_t p = 1;
    for (auto x : v) {
        if (x <= 0) throw std::runtime_error("unflatten: sizes must be positive");
        p *= x;
    }
    return p;
}


static void ensure_all_positive(const Shape& s, const char* what)
{
    for (auto x : s.dims) {
        if (x <= 0)
            throw std::runtime_error(std::string(what) + ": sizes must be positive (no -1 inference)");
    }
}


void validate_unflatten(const Shape& shape, int dim, const Shape& sizes)
{
    int ndim = static_cast<int>(shape.dims.size());
    int d = normalize_dim(dim, ndim);
    if (sizes.dims.empty())
        throw std::runtime_error("unflatten: sizes must have at least 1 dim");
    
    ensure_all_positive(sizes, "unflatten");

    int64_t target = shape.dims[d];
    int64_t p = product(sizes.dims);

    if (p != target) {
        throw std::runtime_error("unflatten: product(sizes) != size of dimension being unflattened. Got " +
                                 std::to_string(p) + " vs " + std::to_string(target));
    }
}


Shape compute_unflatten_shape(const Shape& shape, int dim, const Shape& sizes)
{
    int ndim = static_cast<int>(shape.dims.size());
    int d = normalize_dim(dim, ndim);

    Shape out;
    out.dims.reserve(ndim - 1 + sizes.dims.size());
    for (int i = 0; i < d; ++i) out.dims.push_back(shape.dims[i]);
    for (auto s : sizes.dims)  out.dims.push_back(s);
    for (int i = d + 1; i < ndim; ++i) out.dims.push_back(shape.dims[i]);
    return out;
}


Stride compute_unflatten_strides(const Stride& old_stride, int dim, const Shape& sizes)
{
    // Expand the stride at position dim into a contiguous block consistent with row-major ordering
    // relative to the original step at that dim.
    // Example:
    //   original: shape [A, B, C], stride [sA, sB, sC]
    //   unflatten dim=1 with sizes [U, V] where U*V == B
    //   result stride: [sA, sB*V, sB, sC]
    int64_t step = old_stride.strides[dim];

    // Build the expanded sub-strides: for sizes [d0, d1, ..., dk-1]
    // inner-most has step 1 relative to subspace (i.e., step), so:
    //   sub[k-1] = step
    //   sub[k-2] = step * sizes[k-1]
    //   ...
    // Overall, this ensures the unflattened block is contiguous wrt the original dim’s step.
    std::vector<int64_t> sub(sizes.dims.size(), 0);
    int k = static_cast<int>(sizes.dims.size());
    int64_t cur = step;
    for (int i = k - 1; i >= 0; --i) {
        sub[i] = cur;
        cur *= sizes.dims[i];
    }

    // Construct new stride array
    Stride out;
    out.strides.reserve(old_stride.strides.size() - 1 + sizes.dims.size());
    for (int i = 0; i < dim; ++i) out.strides.push_back(old_stride.strides[i]);
    for (int i = 0; i < k;   ++i) out.strides.push_back(sub[i]);
    for (size_t i = dim + 1; i < old_stride.strides.size(); ++i) out.strides.push_back(old_stride.strides[i]);
    return out;
}
} // namespace ViewUtils
} // namespace OwnTensor