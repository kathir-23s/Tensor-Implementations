#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"

namespace OwnTensor
{   
template<typename T>
__global__ void bool_eq_kernel(const T* a, const T* b, bool* output, size_t n)//✨✨✨  
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Placeholder operation: set output to true if a[idx] equals b[idx], else false
        output[idx] = (a[idx] == b[idx]);
    }
}

template<>
__global__ void bool_eq_kernel<__half>(const __half* a, const __half* b, bool* output, size_t n)//✨✨✨  
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __heq(a[idx],b[idx]);   
    }
}

template<>
__global__ void bool_eq_kernel<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, bool* output, size_t n)//✨✨✨  
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __heq(a[idx],b[idx]);
    }
}

template<typename T>
__global__ void bool_eq_kernel_broadcast(const T* a, const T* b, bool* output, 
                                       size_t a_rows, size_t a_cols,
                                       size_t b_rows, size_t b_cols,
                                       size_t out_rows, size_t out_cols)
{
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
            
            output[idx] = (a[a_idx] == b[b_idx]);
        }
}
template<>
__global__ void bool_eq_kernel_broadcast<__half>(const __half* a, const __half* b, bool* output, 
                                       size_t a_rows, size_t a_cols,
                                       size_t b_rows, size_t b_cols,
                                       size_t out_rows, size_t out_cols)
{
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
            
             output[idx] = __heq(a[a_idx],b[b_idx]);   
        }
}

template<>
__global__ void bool_eq_kernel_broadcast<__nv_bfloat16>(const __nv_bfloat16* a, const  __nv_bfloat16* b, bool* output, 
                                       size_t a_rows, size_t a_cols,
                                       size_t b_rows, size_t b_cols,
                                       size_t out_rows, size_t out_cols)
{
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
            
            output[idx] = __heq(a[a_idx],b[b_idx]);   
        }
}
void cuda_bool_eq_outplace( const Tensor& A, const Tensor& B, Tensor & output, cudaStream_t stream)//✨✨✨  
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
    bool* output_ptr = output.data<bool>();
    
    if (!needs_broadcasting) {
        bool_eq_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, output_ptr ,total_elems);//✨✨✨
    }
    else{
        size_t a_rows = A.shape().dims[0];
        size_t a_cols = A.shape().dims[1];
        size_t b_rows = B.shape().dims[0];
        size_t b_cols = B.shape().dims[1];
        size_t out_rows = output.shape().dims[0];
        size_t out_cols = output.shape().dims[1];

        bool_eq_kernel_broadcast<<<grid_size, block_size, 0, stream>>>( //✨✨✨
                    a_ptr, b_ptr, output_ptr,
                    a_rows, a_cols, b_rows, b_cols, out_rows, out_cols
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
        // }//✨✨✨
    });
}
}
#endif // WITH_CUDA
