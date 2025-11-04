#include "core/Tensor.h"
#include "ops/helpers/TensorOpUtils.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include "device/DeviceCore.h"
#include <driver_types.h>
#include <stdexcept>
#include <functional>

// Testing if makefile can detect this


namespace OwnTensor {
    Tensor operator+(const Tensor& lhs, const Tensor& rhs) 
    {
        // Tensor output(lhs.shape(), lhs.dtype(), lhs.device());
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            // Calculate broadcasted shape
            auto out_rows = std::max(lhs.shape().dims[0], rhs.shape().dims[0]);
            auto out_cols = std::max(lhs.shape().dims[1], rhs.shape().dims[1]);
            auto output_shape = Shape{{out_rows, out_cols}};
        }
        
        Tensor output(output_shape, lhs.dtype(), lhs.device());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_add_tensor(lhs, rhs, output, stream); //✨✨✨
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
        Tensor output(lhs.shape(), lhs.dtype(), lhs.device());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_sub_tensor(lhs, rhs, output, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a - b;  // This lambda gets passed as 'op'
            });
        }
        return output;
    }

    Tensor operator*(const Tensor& lhs, const Tensor& rhs) 
    {
        Tensor output(lhs.shape(), lhs.dtype(), lhs.device());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_mul_tensor(lhs, rhs, output, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a * b;  // This lambda gets passed as 'op'
            });
        }
        return output;
    }

    Tensor operator/(const Tensor& lhs, const Tensor& rhs) 
    {
        Tensor output(lhs.shape(), lhs.dtype(), lhs.device());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_div_tensor(lhs, rhs, output, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else
        {
            apply_binary_operation(lhs, rhs, output, [](auto a, auto b) {
                return a / b;  // This lambda gets passed as 'op'
            });
        }
        return output;
    }

    Tensor operator+=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_add_tensor_inplace(lhs, rhs, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            // Tensor output(lhs.shape(), lhs.dtype(), lhs.device());
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a + b;  // This lambda gets passed as 'op'
            });
        }
        return lhs;
    }

    Tensor operator-=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_sub_tensor_inplace(lhs, rhs, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a - b;  // This lambda gets passed as 'op'
            });
        }
        return lhs;
    }

    Tensor operator*=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_mul_tensor_inplace(lhs, rhs, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a * b;  // This lambda gets passed as 'op'
            });
        }
        return lhs;
    }

    Tensor operator/=(Tensor& lhs, const Tensor& rhs)
    {
        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_div_tensor_inplace(lhs, rhs, stream); //✨✨✨
            #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
            apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
                return a / b;  // This lambda gets passed as 'op'
            });
        }
        return lhs;
    }
} 