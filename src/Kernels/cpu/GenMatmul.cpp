#include "core/Tensor.h"
#include "ops/Kernels.h"
#include "ops/helpers/GenMatmulUtils.h"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <driver_types.h>//✨✨✨
#include "device/DeviceCore.h"//✨✨✨
#ifdef WITH_CUDA
#include "ops/Matmul.cuh"
#endif

namespace OwnTensor 
{
    Tensor matmul(const Tensor& A, const Tensor& B, [[maybe_unused]]cudaStream_t stream)
    {
        // Validate Input Datatypes
        if (A.dtype() != B.dtype())
        {
            throw std::runtime_error("Matmul: Inputs must be of same datatypes");
        }

        const auto& a_dims = A.shape().dims;
        const auto& b_dims = B.shape().dims;  // FIXED: Was A instead of B

        if (a_dims.size() < 2 || b_dims.size() < 2)  // FIXED: Added condition
        {
            throw std::runtime_error("Matmul: Both Tensors must be at least 2 Dimensional");
        }

        // MATRIX MULTIPLICATION COMPATIBILITY: 
        // LAST DIMENSION OF A MUST MATCH SECOND LAST DIMENSION OF B
        if (a_dims[a_dims.size() - 1] != b_dims[b_dims.size() - 2])
        {
            throw std::runtime_error("Incompatible dimensions for Matrix Multiplication");
        }

        // BROADCAST COMPATIBILITY FOR LEADING DIMENSIONS
        size_t a_ndim = a_dims.size();
        size_t b_ndim = b_dims.size();
        size_t max_ndim = std::max(a_ndim, b_ndim);

        std::vector<int64_t> output_dims(max_ndim);

        // Handle batch dimensions (all but last 2)
        for (size_t i = 0; i < max_ndim - 2; ++i) {
            // Get batch dimension from A (right-aligned)
            int64_t a_dim = 1;
            if (i >= max_ndim - a_ndim) {
                size_t a_idx = i - (max_ndim - a_ndim);
                a_dim = a_dims[a_idx];
            }
            
            // Get batch dimension from B (right-aligned)  
            int64_t b_dim = 1;
            if (i >= max_ndim - b_ndim) {
                size_t b_idx = i - (max_ndim - b_ndim);
                b_dim = b_dims[b_idx];
            }
            
            // Check broadcast compatibility
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                throw std::runtime_error("Incompatible batch dimensions for Matrix Multiplication");
            }
            
            output_dims[i] = std::max(a_dim, b_dim);
        }

        // Set matrix dimensions (last 2 dimensions)
        if (a_ndim >= 2) {
            output_dims[max_ndim - 2] = a_dims[a_ndim - 2];
        }
        if (b_ndim >= 2) {
            output_dims[max_ndim - 1] = b_dims[b_ndim - 1];
        }

        // std::cout << "DEBUG: Output shape = [";
        // for (size_t i = 0; i < output_dims.size(); ++i) {
        //     std::cout << output_dims[i];
        //     if (i != output_dims.size() - 1) std::cout << ", ";
        // }
        std::cout << "]" << std::endl;

        Shape output_shape = {output_dims};
        Tensor output(output_shape, A.dtype(), A.device(), A.requires_grad());

        // Device Dispatch
        if (A.device().is_cuda() && B.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
                cuda_matmul(A, B, output, stream);//✨✨✨
            #else
                throw std::runtime_error("Matmul: CUDA support not compiled");
            #endif
        }
        else
        {
            cpu_matmul(A, B, output);
        }

        return output;
    }
}