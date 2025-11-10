#include <stdexcept>
#include "ops/helpers/ConditionalOps.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"

namespace OwnTensor {

// Backend declarations
void cpu_where(const Tensor& condition, const Tensor& input, 
               const Tensor& other, Tensor& out);
void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out);

// Main where implementation
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other) {
    // Validate condition dtype
    if (condition.dtype() != Dtype::Int32 && condition.dtype() != Dtype::Int64) {
        throw std::runtime_error("Condition must be Int32 or Int64 dtype");
    }
    
    // Validate all tensors on same device
    if (condition.device().index != input.device().index || input.device().index != other.device().index) {
        throw std::runtime_error("All tensors must be on the same device");
    }
    
    // For simplicity, require same shape (broadcasting can be added later)
    if (condition.shape() != input.shape() || input.shape() != other.shape()) {
        throw std::runtime_error("All tensors must have the same shape");
    }
    
    // Determine output dtype (promote input and other)
    Dtype output_dtype = input.dtype();
    if (input.dtype() != other.dtype()) {
        // Simple promotion: Float64 > Float32 > Int64 > Int32 > Int16
        if (other.dtype() == Dtype::Float64 || input.dtype() == Dtype::Float64)
            output_dtype = Dtype::Float64;
        else if (other.dtype() == Dtype::Float32 || input.dtype() == Dtype::Float32)
            output_dtype = Dtype::Float32;
    }
    
    // Create output tensor
    Tensor out(input.shape(), output_dtype, input.device(), false);
    
    // Dispatch to backend
    if (condition.device().is_cuda()) {
        cuda_where(condition, input, other, out);
    } else {
        cpu_where(condition, input, other, out);
    }
    
    return out;
}

// Scalar overloads - create full tensors and call main function
Tensor where(const Tensor& condition, double input_scalar, const Tensor& other) {
    // Create tensor filled with scalar value
    Tensor input_tensor(condition.shape(), other.dtype(), condition.device(), false);
    input_tensor.fill(static_cast<float>(input_scalar));
    return where(condition, input_tensor, other);
}

Tensor where(const Tensor& condition, const Tensor& input, double other_scalar) {
    Tensor other_tensor(condition.shape(), input.dtype(), condition.device(), false);
    other_tensor.fill(static_cast<float>(other_scalar));
    return where(condition, input, other_tensor);
}

Tensor where(const Tensor& condition, double input_scalar, double other_scalar) {
    Tensor input_tensor(condition.shape(), Dtype::Float32, condition.device(), false);
    input_tensor.fill(static_cast<float>(input_scalar));
    Tensor other_tensor(condition.shape(), Dtype::Float32, condition.device(), false);
    other_tensor.fill(static_cast<float>(other_scalar));
    return where(condition, input_tensor, other_tensor);
}

} // namespace OwnTensor
