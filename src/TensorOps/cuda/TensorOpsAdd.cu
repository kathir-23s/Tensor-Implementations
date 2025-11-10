#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"

#include <stdio.h>

namespace OwnTensor
{

    template<typename T>
    __global__ void add_kernel(const T* a, const T* b, T* output, size_t n)
    {
        // printf("General Template Add kernel is executed");
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            output[idx] = a[idx] + b[idx];
        }
    }

    template <>
    __global__ void add_kernel<__half>(const __half* a, const __half* b, __half* output, size_t n)
    {
        // printf("Add Kernel of fp16(half) is executed");
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            output[idx] = __hadd(a[idx], b[idx]);
        }
    }

    template <>
    __global__ void add_kernel<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            output[idx] = __hadd(a[idx], b[idx]);
        }
    }

    template<typename T>
    __global__ void add_kernel_broadcast(const T* a, const T* b, T* output, 
                                       size_t a_rows, size_t a_cols,
                                       size_t b_rows, size_t b_cols,
                                       size_t out_rows, size_t out_cols)
    {        
        // printf("General Template Add kernel broadcast is executed");

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elems = out_rows * out_cols;
        
        if (idx < total_elems) {
            // Convert linear index to 2D coordinates
            size_t i = idx / out_cols;
            size_t j = idx % out_cols;
            
            // Calculate strides for broadcasting
            size_t a_row_stride = (a_rows == 1) ? 0 : a_cols;
            size_t a_col_stride = (a_cols == 1) ? 0 : 1;
            size_t b_row_stride = (b_rows == 1) ? 0 : b_cols;
            size_t b_col_stride = (b_cols == 1) ? 0 : 1;
            
            // Calculate source indices using strides
            size_t a_idx = (i * a_row_stride) + (j * a_col_stride);
            size_t b_idx = (i * b_row_stride) + (j * b_col_stride);
            
            output[idx] = a[a_idx] + b[b_idx];
        }
    }

    template <>
    __global__ void add_kernel_broadcast<__half>(const __half* a, const __half* b, __half* output, 
                                               size_t a_rows, size_t a_cols,
                                               size_t b_rows, size_t b_cols,
                                               size_t out_rows, size_t out_cols)
    {
        // printf("Add Kernel of fp16(half) broadcast is executed");
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elems = out_rows * out_cols;
        
        if (idx < total_elems) {
            size_t i = idx / out_cols;
            size_t j = idx % out_cols;
            
            size_t a_row_stride = (a_rows == 1) ? 0 : a_cols;
            size_t a_col_stride = (a_cols == 1) ? 0 : 1;
            size_t b_row_stride = (b_rows == 1) ? 0 : b_cols;
            size_t b_col_stride = (b_cols == 1) ? 0 : 1;
            
            size_t a_idx = (i * a_row_stride) + (j * a_col_stride);
            size_t b_idx = (i * b_row_stride) + (j * b_col_stride);
            
            output[idx] = __hadd(a[a_idx], b[b_idx]);
        }
    }

    template <>
    __global__ void add_kernel_broadcast<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, 
                                                      size_t a_rows, size_t a_cols,
                                                      size_t b_rows, size_t b_cols,
                                                      size_t out_rows, size_t out_cols)
    {   
        // printf("Add Kernel of bf16(nv_bfloat16) broadcast is executed");

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elems = out_rows * out_cols;
        
        if (idx < total_elems) {
            size_t i = idx / out_cols;
            size_t j = idx % out_cols;
            
            size_t a_row_stride = (a_rows == 1) ? 0 : a_cols;
            size_t a_col_stride = (a_cols == 1) ? 0 : 1;
            size_t b_row_stride = (b_rows == 1) ? 0 : b_cols;
            size_t b_col_stride = (b_cols == 1) ? 0 : 1;
            
            size_t a_idx = (i * a_row_stride) + (j * a_col_stride);
            size_t b_idx = (i * b_row_stride) + (j * b_col_stride);
            
            output[idx] = __hadd(a[a_idx], b[b_idx]);
        }
    }

    void cuda_add_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream) //✨✨✨
    {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = output.numel();
        size_t block_size = 256;
        size_t grid_size = (total_elems + block_size - 1) / block_size;
        
       // std::cout << "Addition CUDA\n";
        
        dispatch_by_dtype(A.dtype(), [&](auto dummy)
        {
            using T = decltype(dummy);
            const T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            T* output_ptr = output.data<T>();
            
            if (!needs_broadcasting) {
                // Original same-shape kernel
                add_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, output_ptr, total_elems); //✨✨✨
            } else {
                // New broadcast kernel with shape information
                size_t a_rows = A.shape().dims[0];
                size_t a_cols = A.shape().dims[1];
                size_t b_rows = B.shape().dims[0];
                size_t b_cols = B.shape().dims[1];
                size_t out_rows = output.shape().dims[0];
                size_t out_cols = output.shape().dims[1];
                
                add_kernel_broadcast<<<grid_size, block_size, 0, stream>>>( //✨✨✨
                    a_ptr, b_ptr, output_ptr,
                    a_rows, a_cols, b_rows, b_cols, out_rows, out_cols
                );
            }
             //✨✨✨ removed for asynch purpose dont uncommit
            // cudaError_t err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     throw std::runtime_error("Addition CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
            // }
            
            // err = cudaDeviceSynchronize();
            // if (err != cudaSuccess) {
            //     throw std::runtime_error("Addition CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
            // } //✨✨✨
        });
    }

/*########################################################
            TENSOR INPLACE CUDA KERNELS
##########################################################*/

    template <typename T>
    __global__ void add_inplace_kernel(T* lhs, const T* rhs, size_t n)
    {        
        // printf("General Template Add kernel inplace is executed");

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] += rhs[idx];
        }
    }

    template <>
    __global__ void add_inplace_kernel<__half>(__half* lhs, const __half* rhs, size_t n)
    {
        // printf("Add Kernel of fp16(half) inplace is executed");
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] = __hadd(lhs[idx], rhs[idx]);
        }
    }

    template <>
    __global__ void add_inplace_kernel<__nv_bfloat16>(__nv_bfloat16* lhs, const __nv_bfloat16* rhs, size_t n)
    {
        // printf("Add Kernel of bf16(nv_bfloat16) inplace is executed");

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] = __hadd(lhs[idx],rhs[idx]);
        }
    }

    template <typename T>
    __global__ void add_inplace_kernel_broadcast(T* lhs, const T* rhs,
                                               size_t lhs_rows, size_t lhs_cols,
                                               size_t rhs_rows, size_t rhs_cols,
                                               size_t out_rows, size_t out_cols)
    {        
        // printf("General Template Add kernel inplace broadcast is executed");

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elems = out_rows * out_cols;
        
        if (idx < total_elems) {
            size_t i = idx / out_cols;
            size_t j = idx % out_cols;
            
            size_t lhs_row_stride = (lhs_rows == 1) ? 0 : lhs_cols;
            size_t lhs_col_stride = (lhs_cols == 1) ? 0 : 1;
            size_t rhs_row_stride = (rhs_rows == 1) ? 0 : rhs_cols;
            size_t rhs_col_stride = (rhs_cols == 1) ? 0 : 1;
            
            size_t lhs_idx = (i * lhs_row_stride) + (j * lhs_col_stride);
            size_t rhs_idx = (i * rhs_row_stride) + (j * rhs_col_stride);
            
            lhs[lhs_idx] += rhs[rhs_idx];
        }
    }

    template <>
    __global__ void add_inplace_kernel_broadcast<__half>(__half* lhs, const __half* rhs,
                                                       size_t lhs_rows, size_t lhs_cols,
                                                       size_t rhs_rows, size_t rhs_cols,
                                                       size_t out_rows, size_t out_cols)
    {        
        // printf("Add Kernel of fp16(half) inplace broadcast is executed");
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elems = out_rows * out_cols;
        
        if (idx < total_elems) {
            size_t i = idx / out_cols;
            size_t j = idx % out_cols;
            
            size_t lhs_row_stride = (lhs_rows == 1) ? 0 : lhs_cols;
            size_t lhs_col_stride = (lhs_cols == 1) ? 0 : 1;
            size_t rhs_row_stride = (rhs_rows == 1) ? 0 : rhs_cols;
            size_t rhs_col_stride = (rhs_cols == 1) ? 0 : 1;
            
            size_t lhs_idx = (i * lhs_row_stride) + (j * lhs_col_stride);
            size_t rhs_idx = (i * rhs_row_stride) + (j * rhs_col_stride);
            
            lhs[lhs_idx] = __hadd(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    template <>
    __global__ void add_inplace_kernel_broadcast<__nv_bfloat16>(__nv_bfloat16* lhs, const __nv_bfloat16* rhs,
                                                              size_t lhs_rows, size_t lhs_cols,
                                                              size_t rhs_rows, size_t rhs_cols,
                                                              size_t out_rows, size_t out_cols)
    {
        // printf("Add Kernel of bf16(nv_bfloat16) inplace broadcast is executed");

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elems = out_rows * out_cols;
        
        if (idx < total_elems) {
            size_t i = idx / out_cols;
            size_t j = idx % out_cols;
            
            size_t lhs_row_stride = (lhs_rows == 1) ? 0 : lhs_cols;
            size_t lhs_col_stride = (lhs_cols == 1) ? 0 : 1;
            size_t rhs_row_stride = (rhs_rows == 1) ? 0 : rhs_cols;
            size_t rhs_col_stride = (rhs_cols == 1) ? 0 : 1;
            
            size_t lhs_idx = (i * lhs_row_stride) + (j * lhs_col_stride);
            size_t rhs_idx = (i * rhs_row_stride) + (j * rhs_col_stride);
            
            lhs[lhs_idx] = __hadd(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    void cuda_add_tensor_inplace(Tensor& A, const Tensor& B, cudaStream_t stream) //✨✨✨
    {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = A.numel();
        size_t block_size = 256;
        size_t grid_size = (total_elems + block_size - 1) / block_size;

        //std::cout << "Addition Inplace CUDA\n";

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
        {
            using T = decltype(dummy);
            T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            
            if (!needs_broadcasting) {
                add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, total_elems); //✨✨✨
            } else {
                size_t a_rows = A.shape().dims[0];
                size_t a_cols = A.shape().dims[1];
                size_t b_rows = B.shape().dims[0];
                size_t b_cols = B.shape().dims[1];
                size_t out_rows = A.shape().dims[0];
                size_t out_cols = A.shape().dims[1];
                
                add_inplace_kernel_broadcast<<<grid_size, block_size, 0, stream>>>( //✨✨✨
                    a_ptr, b_ptr, a_rows, a_cols, b_rows, b_cols, out_rows, out_cols
                );
            }
             //✨✨✨
            // cudaError_t err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     throw std::runtime_error("Addition CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
            // }
            
            // err = cudaDeviceSynchronize();
            // if (err != cudaSuccess) {
            //     throw std::runtime_error("Addition CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
            // } //✨✨✨
        });
    }

}
#endif