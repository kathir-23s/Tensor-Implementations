#include "core/Tensor.h"
#include "ops/helpers/TensorOpUtils.h"
#include "ops/helpers/BroadcastUtils.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include "device/DeviceCore.h"
#include <driver_types.h>
#include <stdexcept>
#include <functional>

namespace OwnTensor {
    Tensor operator+(const Tensor& lhs, const Tensor& rhs) 
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, lhs.dtype(), lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_add_tensor(lhs, rhs, output, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a + b;
            });
        }
        return output;
    }

    Tensor operator-(const Tensor& lhs, const Tensor& rhs) 
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, lhs.dtype(), lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_sub_tensor(lhs, rhs, output, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a - b;
            });
        }
        return output;
    }

    Tensor operator*(const Tensor& lhs, const Tensor& rhs) 
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, lhs.dtype(), lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_mul_tensor(lhs, rhs, output, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a * b;
            });
        }
        return output;
    }

    Tensor operator/(const Tensor& lhs, const Tensor& rhs) 
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, lhs.dtype(), lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_div_tensor(lhs, rhs, output, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a / b;
            });
        }
        return output;
    }

    // =================================================================
    // IN-PLACE OPERATORS WITH VALIDATION
    // =================================================================

    Tensor operator+=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.shape().dims != rhs.shape().dims) {
            Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
            if (lhs.shape().dims != broadcasted.dims) {
                throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast ");
            }
        }

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_add_tensor_inplace(lhs, rhs, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a + b;
            });
        }
        return lhs;
    }

    Tensor operator-=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.shape().dims != rhs.shape().dims) {
            Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
            if (lhs.shape().dims != broadcasted.dims) {
                throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast ");
            }
        }

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_sub_tensor_inplace(lhs, rhs, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a - b;
            });
        }
        return lhs;
    }

    Tensor operator*=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.shape().dims != rhs.shape().dims) {
            Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
            if (lhs.shape().dims != broadcasted.dims) {
                throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast ");
            }
        }

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_mul_tensor_inplace(lhs, rhs, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a * b;
            });
        }
        return lhs;
    }

    Tensor operator/=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.shape().dims != rhs.shape().dims) {
            Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
            if (lhs.shape().dims != broadcasted.dims) {
                throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast ");
            }
        }

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_div_tensor_inplace(lhs, rhs, stream);
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a / b;
            });
        }
        return lhs;
    }
}
