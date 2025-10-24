#include "core/Views/ViewUtils.h"
#include "core/Tensor.h"
#include <stdexcept>

namespace OwnTensor {
// Implementation of Tensor::view()
Tensor Tensor::view(Shape new_shape) const {
    // Check if tensor is contiguous
    if (!is_contiguous()) {
        throw std::runtime_error(
            "view() requires contiguous tensor. "
            "Use reshape() or call contiguous() first."
        );
    }
    // Handle -1 dimension inference
    ViewUtils::infer_dimension(numel(), new_shape);
    // Validate that total elements match
    if (!ViewUtils::is_shape_compatible(numel(), new_shape)) {
        throw std::runtime_error(
            "view: Shape mismatch - new shape must have same number of elements"
        );
    }
    // Compute new strides for the new shape
    Stride new_stride = ViewUtils::compute_strides(new_shape);
    // Create and return view tensor (shares data_ptr_)
    return Tensor(data_ptr_,           // Share the data pointer
                  new_shape,            // New shape
                  new_stride,           // New strides
                  storage_offset_,      // Same offset
                  dtype_,               // Same dtype
                  device_,              // Same device
                  requires_grad_);      // Same requires_grad
}

Tensor Tensor::reshape(Shape new_shape) const {
    // Infer any -1 dimension
    ViewUtils::infer_dimension(numel(), new_shape);
    // Validate element count
    if (!ViewUtils::is_shape_compatible(numel(), new_shape)) {
        throw std::runtime_error("reshape: new shape has different number of elements");
    }
    // if contiguous, identical to view()
    if (is_contiguous()) {
        Stride new_stride = ViewUtils::compute_strides(new_shape);
        return Tensor(
            data_ptr_, new_shape, new_stride, storage_offset_,
            dtype_, device_, requires_grad_
        );
    }
    // materialize contiguous copy on same device then view it
    Tensor base = contiguous();  // contiguous() returns row-major layout
    Stride new_stride = ViewUtils::compute_strides(new_shape);
    return Tensor(
        base.data_ptr_, new_shape, new_stride, base.storage_offset_,
        base.dtype_, base.device_, base.requires_grad_
    );
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
    int ndim = static_cast<int>(shape_.dims.size());
    dim0 = ViewUtils::normalize_dim(dim0, ndim);
    dim1 = ViewUtils::normalize_dim(dim1, ndim);
    if (dim0 == dim1) {
        // No-op: return a view with identical metadata
        return Tensor(
            data_ptr_, shape_, stride_, storage_offset_,
            dtype_, device_, requires_grad_
        );
    }

    Shape new_shape = shape_;
    Stride new_stride = stride_;
    ViewUtils::swap_dimensions(new_shape, new_stride, dim0, dim1);

    return Tensor(
        data_ptr_, new_shape, new_stride, storage_offset_,
        dtype_, device_, requires_grad_
    );
}

Tensor Tensor::t() const 
{ if (shape_.dims.size() < 2) return *this; return transpose(-2, -1); }

Tensor Tensor::flatten(int start_dim, int end_dim) const
{
    int ndim = static_cast<int>(shape_.dims.size());
    start_dim = ViewUtils::normalize_dim(start_dim, ndim);
    end_dim   = ViewUtils::normalize_dim(end_dim < 0 ? end_dim + ndim : end_dim, ndim);
    if (start_dim > end_dim) {
        throw std::runtime_error("flatten: start_dim must be <= end_dim");
    }

    // Will throw if any participating dims are non-positive
    Shape new_shape = ViewUtils::compute_flatten_shape(shape_, start_dim, end_dim);

    if (is_contiguous()) {
        Stride new_stride = ViewUtils::compute_strides(new_shape);
        return Tensor(
            data_ptr_, new_shape, new_stride, storage_offset_,
            dtype_, device_, requires_grad_
        );
    } else {
        Tensor base = contiguous();
        Stride new_stride = ViewUtils::compute_strides(new_shape);
        return Tensor(
            base.data_ptr_, new_shape, new_stride, base.storage_offset_,
            base.dtype_, base.device_, base.requires_grad_
        );
    }
}

Tensor Tensor::unflatten(int dim, Shape sizes) const
{
    int ndim = static_cast<int>(shape_.dims.size());
    dim = ViewUtils::normalize_dim(dim, ndim);

    // Validates: sizes positive, product equals shape_[dim]
    ViewUtils::validate_unflatten(shape_, dim, sizes);

    Shape new_shape  = ViewUtils::compute_unflatten_shape(shape_,  dim, sizes);
    Stride new_stride = ViewUtils::compute_unflatten_strides(stride_, dim, sizes);

    return Tensor(
        data_ptr_, new_shape, new_stride, storage_offset_,
        dtype_, device_, requires_grad_
    );
}

} // namespace OwnTensor