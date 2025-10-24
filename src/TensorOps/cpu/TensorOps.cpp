#include "core/Tensor.h"
#include "ops/TensorOpUtils.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include <stdexcept>
#include <functional>

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
                cuda_add_tensor(lhs, rhs, output);
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
                cuda_sub_tensor(lhs, rhs, output);
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
                cuda_mul_tensor(lhs, rhs, output);
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
                cuda_div_tensor(lhs, rhs, output);
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
        // Tensor output(lhs.shape(), lhs.dtype(), lhs.device());
        apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
            return a + b;  // This lambda gets passed as 'op'
        });
        return lhs;
    }

    Tensor operator-=(Tensor& lhs, const Tensor& rhs)
    {
        apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
            return a - b;  // This lambda gets passed as 'op'
        });
        return lhs;
    }

    Tensor operator*=(Tensor& lhs, const Tensor& rhs)
    {
        apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
            return a * b;  // This lambda gets passed as 'op'
        });
        return lhs;
    }

    Tensor operator/=(Tensor& lhs, const Tensor& rhs)
    {
        apply_binary_operation(lhs, rhs, lhs, [](auto a, auto b) {
            return a / b;  // This lambda gets passed as 'op'
        });
        return lhs;
    }
} 