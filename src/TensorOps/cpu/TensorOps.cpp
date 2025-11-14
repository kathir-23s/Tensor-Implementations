#include "core/Tensor.h"
#include "ops/helpers/TensorOpUtils.h"
#include "ops/helpers/BroadcastUtils.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include "device/DeviceCore.h"
#include "dtype/DtypeTraits.h"  // ✅ ADD THIS for promote_dtypes_bool
#include <driver_types.h>
#include <stdexcept>
#include <functional>

namespace OwnTensor {

// ============================================================================
// HELPER: Convert tensor to promoted dtype if needed
// ============================================================================
static Tensor promote_if_needed(const Tensor& input, Dtype target_dtype) {
    if (input.dtype() == target_dtype) {
        return input;  // No conversion needed
    }
    
    // Convert to target dtype
    return input.as_type(target_dtype);
}

// ============================================================================
// ADDITION OPERATOR
// ============================================================================
Tensor operator+(const Tensor& lhs, const Tensor& rhs) 
{
    // ✅ 1. Determine promoted dtype
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    
    // ✅ 2. Convert operands if needed
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    // ✅ 3. Compute output shape (broadcasting)
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    // ✅ 4. Create output tensor with promoted dtype
    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    // ✅ 5. Perform operation
    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_add_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a + b;
        });
    }
    return output;
}

// ============================================================================
// SUBTRACTION OPERATOR
// ============================================================================
Tensor operator-(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_sub_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a - b;
        });
    }
    return output;
}

// ============================================================================
// MULTIPLICATION OPERATOR
// ============================================================================
Tensor operator*(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_mul_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a * b;
        });
    }
    return output;
}

// ============================================================================
// DIVISION OPERATOR
// ============================================================================
Tensor operator/(const Tensor& lhs, const Tensor& rhs) 
{
    // ✅ USE DIVISION-SPECIFIC PROMOTION
    Dtype promoted_dtype = promote_dtypes_division(lhs.dtype(), rhs.dtype());
    
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_div_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a / b;
        });
    }
    return output;
}
// ============================================================================
// IN-PLACE OPERATORS (NO TYPE PROMOTION ALLOWED)
// ============================================================================

Tensor operator+=(Tensor& lhs, const Tensor& rhs)
{
    // ⚠️ In-place ops require compatible types (rhs must be promotable to lhs)
    if (lhs.dtype() != rhs.dtype()) {
        // Check if rhs can be safely promoted to lhs type
        Dtype promoted = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
        if (promoted != lhs.dtype()) {
            throw std::runtime_error(
                "In-place operator +=: Cannot promote result type beyond lhs dtype. "
                "LHS: " + get_dtype_name(lhs.dtype()) + ", RHS: " + get_dtype_name(rhs.dtype())
            );
        }
    }
    
    // Convert rhs to lhs dtype if needed
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_add_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a + b;
        });
    }
    return lhs;
}

Tensor operator-=(Tensor& lhs, const Tensor& rhs)
{
    if (lhs.dtype() != rhs.dtype()) {
        Dtype promoted = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
        if (promoted != lhs.dtype()) {
            throw std::runtime_error(
                "In-place operator -=: Cannot promote result type beyond lhs dtype. "
                "LHS: " + get_dtype_name(lhs.dtype()) + ", RHS: " + get_dtype_name(rhs.dtype())
            );
        }
    }
    
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_sub_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a - b;
        });
    }
    return lhs;
}

Tensor operator*=(Tensor& lhs, const Tensor& rhs)
{
    if (lhs.dtype() != rhs.dtype()) {
        Dtype promoted = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
        if (promoted != lhs.dtype()) {
            throw std::runtime_error(
                "In-place operator *=: Cannot promote result type beyond lhs dtype. "
                "LHS: " + get_dtype_name(lhs.dtype()) + ", RHS: " + get_dtype_name(rhs.dtype())
            );
        }
    }
    
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_mul_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a * b;
        });
    }
    return lhs;
}

Tensor operator/=(Tensor& lhs, const Tensor& rhs)
{
    // ✅ Check if division would require float promotion
    Dtype div_promoted = promote_dtypes_division(lhs.dtype(), rhs.dtype());
    
    if (div_promoted != lhs.dtype()) {
        throw std::runtime_error(
            "In-place operator /=: Division requires float promotion. "
            "LHS: " + get_dtype_name(lhs.dtype()) + 
            " cannot store result of type " + get_dtype_name(div_promoted) + ". "
            "Use regular division (/) instead."
        );
    }
    
    // Rest of the code remains the same...
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_div_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a / b;
        });
    }
    return lhs;
}
   // ============================================================================
// COMPARISON OPERATORS WITH TYPE PROMOTION (Fixed Version)
// ============================================================================

Tensor operator==(const Tensor& lhs, const Tensor& rhs) 
{
    // ✅ 1. Promote types to common dtype (use arithmetic promotion rules)
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    
    // ✅ 2. Convert operands if needed
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    // ✅ 3. Compute output shape (broadcasting)
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    // ✅ 4. Create output tensor (always Bool dtype)
    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    // ✅ 5. Perform comparison using promoted types
    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_eq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a == b;
        });
    }
    return output;
}

Tensor operator!=(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_neq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a != b;
        });
    }
    return output;
}

Tensor operator<=(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_leq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a <= b;
        });
    }
    return output;
}

Tensor operator>=(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_geq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a >= b;
        });
    }
    return output;
}

Tensor operator>(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_gt_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a > b;
        });
    }
    return output;
}

Tensor operator<(const Tensor& lhs, const Tensor& rhs) 
{
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_lt_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a < b;
        });
    }
    return output;
}

    Tensor logical_AND(const Tensor& lhs, const Tensor& rhs)
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_logical_and_outplace(lhs, rhs,output, stream); //✨✨✨
           #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
        apply_binary_op_bool(lhs, rhs, output, [](auto a, auto b) {
            return a && b;  // This lambda gets passed as 'op'
        });
        }
        return output;

    }

    Tensor logical_OR(const Tensor& lhs, const Tensor& rhs)
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_logical_or_outplace(lhs, rhs,output, stream); //✨✨✨
           #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
        apply_binary_op_bool(lhs, rhs, output, [](auto a, auto b) {
            return a || b;  // This lambda gets passed as 'op'
        });
        }
        return output;
        
    }

    Tensor logical_XOR(const Tensor& lhs, const Tensor& rhs)
{
    Shape output_shape = lhs.shape();
    if (lhs.shape().dims != rhs.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            cuda_logical_xor_outplace(lhs, rhs, output, stream);//✨✨✨
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        // ✅ FIXED: Convert to bool first, then XOR
        apply_binary_op_bool(lhs, rhs, output, [](auto a, auto b) {
            // Convert to boolean (non-zero = true), then XOR
            bool a_bool = (a != decltype(a)(0));
            bool b_bool = (b != decltype(b)(0));
            return a_bool != b_bool;  // XOR is "not equal" of boolean values
        });
    }
    return output;
}
    Tensor logical_NOT(const Tensor& lhs)
    {
        Shape output_shape = lhs.shape();
        
        Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_logical_not_outplace(lhs,output, stream); //✨✨✨
           #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
        apply_not_bool(lhs, output, [](auto a) {
            return !a ;  // This lambda gets passed as 'op'
        });
        }
        return output;
        
    }
}
