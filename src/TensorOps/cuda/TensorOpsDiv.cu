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
__global__ void div_kernel(const T* a, const T* b, T* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = a[idx] / b[idx];
    }
}

template <>
__global__ void div_kernel<__half>(const __half* a, const __half* b, __half* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __hdiv(a[idx], b[idx]);
    }
}

template <>
__global__ void div_kernel<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __hdiv(a[idx],b[idx]);
    }
}

template<typename T>
__global__ void div_kernel_nd_broadcast(const T* a, const T* b, T* output,
                                      const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                      const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                      size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                      size_t total_elems)
{
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total_elems) return;

    size_t a_bcast_strides[8] = {0};
    size_t b_bcast_strides[8] = {0};

    // Pre-calculate broadcast strides
    for (size_t i = 0; i < out_ndim; ++i) {
        size_t a_dim_idx = a_ndim > i ? a_ndim - 1 - i : 0;
        size_t b_dim_idx = b_ndim > i ? b_ndim - 1 - i : 0;
        size_t out_dim_idx = out_ndim - 1 - i;

        if (a_ndim > i && a_shape[a_dim_idx] == out_shape[out_dim_idx]) {
            a_bcast_strides[out_dim_idx] = a_strides[a_dim_idx];
        }

        if (b_ndim > i && b_shape[b_dim_idx] == out_shape[out_dim_idx]) {
            b_bcast_strides[out_dim_idx] = b_strides[b_dim_idx];
        }
    }

    size_t coords[8];
    size_t temp_idx = linear_idx;
    for (int dim = out_ndim - 1; dim >= 0; --dim) {
        coords[dim] = temp_idx % out_shape[dim];
        temp_idx /= out_shape[dim];
    }

    size_t a_idx = 0;
    size_t b_idx = 0;
    for (size_t dim = 0; dim < out_ndim; ++dim) {
        a_idx += coords[dim] * a_bcast_strides[dim];
        b_idx += coords[dim] * b_bcast_strides[dim];
    }

    output[linear_idx] = a[a_idx] / b[b_idx];
}

template<>
__global__ void div_kernel_nd_broadcast<__half>(const __half* a, const __half* b, __half* output,
                                              const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                              const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                              size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                              size_t total_elems)
{
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total_elems) return;

    size_t a_bcast_strides[8] = {0};
    size_t b_bcast_strides[8] = {0};

    for (size_t i = 0; i < out_ndim; ++i) {
        size_t a_dim_idx = a_ndim > i ? a_ndim - 1 - i : 0;
        size_t b_dim_idx = b_ndim > i ? b_ndim - 1 - i : 0;
        size_t out_dim_idx = out_ndim - 1 - i;

        if (a_ndim > i && a_shape[a_dim_idx] == out_shape[out_dim_idx]) {
            a_bcast_strides[out_dim_idx] = a_strides[a_dim_idx];
        }

        if (b_ndim > i && b_shape[b_dim_idx] == out_shape[out_dim_idx]) {
            b_bcast_strides[out_dim_idx] = b_strides[b_dim_idx];
        }
    }

    size_t coords[8];
    size_t temp_idx = linear_idx;
    for (int dim = out_ndim - 1; dim >= 0; --dim) {
        coords[dim] = temp_idx % out_shape[dim];
        temp_idx /= out_shape[dim];
    }

    size_t a_idx = 0;
    size_t b_idx = 0;
    for (size_t dim = 0; dim < out_ndim; ++dim) {
        a_idx += coords[dim] * a_bcast_strides[dim];
        b_idx += coords[dim] * b_bcast_strides[dim];
    }

    output[linear_idx] = __hdiv(a[a_idx], b[b_idx]);
}

template<>
__global__ void div_kernel_nd_broadcast<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output,
                                                    const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                                    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                                    size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                                    size_t total_elems)
{
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total_elems) return;

    size_t a_bcast_strides[8] = {0};
    size_t b_bcast_strides[8] = {0};

    for (size_t i = 0; i < out_ndim; ++i) {
        size_t a_dim_idx = a_ndim > i ? a_ndim - 1 - i : 0;
        size_t b_dim_idx = b_ndim > i ? b_ndim - 1 - i : 0;
        size_t out_dim_idx = out_ndim - 1 - i;

        if (a_ndim > i && a_shape[a_dim_idx] == out_shape[out_dim_idx]) {
            a_bcast_strides[out_dim_idx] = a_strides[a_dim_idx];
        }

        if (b_ndim > i && b_shape[b_dim_idx] == out_shape[out_dim_idx]) {
            b_bcast_strides[out_dim_idx] = b_strides[b_dim_idx];
        }
    }

    size_t coords[8];
    size_t temp_idx = linear_idx;
    for (int dim = out_ndim - 1; dim >= 0; --dim) {
        coords[dim] = temp_idx % out_shape[dim];
        temp_idx /= out_shape[dim];
    }

    size_t a_idx = 0;
    size_t b_idx = 0;
    for (size_t dim = 0; dim < out_ndim; ++dim) {
        a_idx += coords[dim] * a_bcast_strides[dim];
        b_idx += coords[dim] * b_bcast_strides[dim];
    }

    output[linear_idx] = __hdiv(a[a_idx], b[b_idx]);
}

void cuda_div_tensor(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
{
    bool needs_broadcasting = (A.shape().dims != B.shape().dims);
    size_t total_elems = output.numel();
    size_t block_size = 256;
    size_t grid_size = (total_elems + block_size - 1) / block_size;
    
    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        const T* b_ptr = B.data<T>();
        T* output_ptr = output.data<T>();
        
        if (!needs_broadcasting) {
            div_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, output_ptr, total_elems);
        } else {
            const auto& a_shape = A.shape().dims;
            const auto& b_shape = B.shape().dims;
            const auto& out_shape = output.shape().dims;
            
            size_t a_ndim = a_shape.size();
            size_t b_ndim = b_shape.size();
            size_t out_ndim = out_shape.size();
            
            size_t *d_a_shape, *d_b_shape, *d_out_shape;
            size_t *d_a_strides, *d_b_strides, *d_out_strides;
            
            cudaMalloc(&d_a_shape, a_ndim * sizeof(size_t));
            cudaMalloc(&d_b_shape, b_ndim * sizeof(size_t));
            cudaMalloc(&d_out_shape, out_ndim * sizeof(size_t));
            cudaMalloc(&d_a_strides, a_ndim * sizeof(size_t));
            cudaMalloc(&d_b_strides, b_ndim * sizeof(size_t));
            cudaMalloc(&d_out_strides, out_ndim * sizeof(size_t));
            
            cudaMemcpyAsync(d_a_shape, a_shape.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_b_shape, b_shape.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_out_shape, out_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_a_strides, A.stride().strides.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_b_strides, B.stride().strides.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_out_strides, output.stride().strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            
            div_kernel_nd_broadcast<<<grid_size, block_size, 0, stream>>>(
                a_ptr, b_ptr, output_ptr,
                d_a_shape, d_b_shape, d_out_shape,
                d_a_strides, d_b_strides, d_out_strides,
                a_ndim, b_ndim, out_ndim, total_elems
            );
            
            cudaFree(d_a_shape);
            cudaFree(d_b_shape);
            cudaFree(d_out_shape);
            cudaFree(d_a_strides);
            cudaFree(d_b_strides);
            cudaFree(d_out_strides);
        }
    });
}

/*########################################################
            TENSOR INPLACE CUDA KERNELS
##########################################################*/


template <typename T>
    __global__ void div_inplace_kernel(T* lhs, const T* rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] /= rhs[idx];
        }
    }

    template <>
    __global__ void div_inplace_kernel<__half>(__half* lhs, const __half* rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] = __hdiv(lhs[idx], rhs[idx]);
        }
    }

    template <>
    __global__ void div_inplace_kernel<__nv_bfloat16>(__nv_bfloat16* lhs, const __nv_bfloat16* rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] = __hdiv(lhs[idx],rhs[idx]);
        }
    }

    template <typename T>
    __global__ void div_inplace_kernel_broadcast(T* lhs, const T* rhs,
                                               size_t lhs_rows, size_t lhs_cols,
                                               size_t rhs_rows, size_t rhs_cols,
                                               size_t out_rows, size_t out_cols)
    {
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
            
            lhs[lhs_idx] /= rhs[rhs_idx];
        }
    }

    template <>
    __global__ void div_inplace_kernel_broadcast<__half>(__half* lhs, const __half* rhs,
                                                       size_t lhs_rows, size_t lhs_cols,
                                                       size_t rhs_rows, size_t rhs_cols,
                                                       size_t out_rows, size_t out_cols)
    {
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
            
            lhs[lhs_idx] = __hdiv(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    template <>
    __global__ void div_inplace_kernel_broadcast<__nv_bfloat16>(__nv_bfloat16* lhs, const __nv_bfloat16* rhs,
                                                              size_t lhs_rows, size_t lhs_cols,
                                                              size_t rhs_rows, size_t rhs_cols,
                                                              size_t out_rows, size_t out_cols)
    {
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
            
            lhs[lhs_idx] = __hdiv(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    void cuda_div_tensor_inplace(Tensor& A, const Tensor& B, cudaStream_t stream)
    {
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = A.numel();
        size_t block_size = 256;
        size_t grid_size = (total_elems + block_size - 1) / block_size;

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
        {
            using T = decltype(dummy);
            T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            
            if (!needs_broadcasting) {
                div_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, total_elems);
            } else {
                size_t a_rows = A.shape().dims[0];
                size_t a_cols = A.shape().dims[1];
                size_t b_rows = B.shape().dims[0];
                size_t b_cols = B.shape().dims[1];
                size_t out_rows = A.shape().dims[0];
                size_t out_cols = A.shape().dims[1];
                
                div_inplace_kernel_broadcast<<<grid_size, block_size, 0, stream>>>(
                    a_ptr, b_ptr, a_rows, a_cols, b_rows, b_cols, out_rows, out_cols
                );
            }

        });
    }


}

#endif