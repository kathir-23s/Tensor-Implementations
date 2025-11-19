#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

#include "ops/Matmul.cuh"
#include "core/Tensor.h"

namespace OwnTensor {

    template <typename T>
    __global__ void batched_matmul_kernel(const T* A, const T* B, T* output,
                                        const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                        const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                        size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                        size_t total_batches)
    {
        size_t batch_idx = blockIdx.z;
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        // Matrix dimensions
        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        if (i >= m || j >= p) return;

        // Calculate batch offsets
        // Calculate batch offsets
        size_t a_batch_offset = 0;
        size_t b_batch_offset = 0;
        size_t out_batch_offset = 0;

        // size_t temp_batch = batch_idx;
        // for (int dim = out_ndim - 3; dim >= 0; --dim) {
            //     size_t batch_dim_size = out_shape[dim];
            //     size_t batch_coord = temp_batch % batch_dim_size;
            //     temp_batch /= batch_dim_size;
            
            //     // Calculate offsets using the actual batch coordinates
            //     if (dim < a_ndim - 2) {
                //         a_batch_offset += batch_coord * a_strides[dim];
                //     }
                //     if (dim < b_ndim - 2) {
                    //         b_batch_offset += batch_coord * b_strides[dim];
                    //     }
                    //     out_batch_offset += batch_coord * out_strides[dim];
                    // }
                    
        // FIXED: Proper batch offset calculation
        size_t temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim)
        {
            size_t batch_dim_size = out_shape[dim];
            size_t batch_coord = temp_batch % batch_dim_size;
            temp_batch /= batch_dim_size;

            out_batch_offset += batch_coord * out_strides[dim];

            // RIGHT-ALIGNED: Calculating corresponding dimensions for A and B
            size_t a_corres_dim = dim - (out_ndim - 2 - (a_ndim - 2));
            size_t b_corres_dim = dim - (out_ndim - 2 - (b_ndim - 2));

            // For A and B: Right aligned broadcasting rules
            
            if (dim >= out_ndim - 2 - (a_ndim - 2))
            {
                size_t a_dim_size = a_shape[a_corres_dim];
                size_t a_idx = (a_dim_size > 1) ? batch_coord : 0;
                a_batch_offset += a_idx * a_strides[a_corres_dim];
            }

            if (dim >= out_ndim - 2 - (b_ndim - 2))
            {
                size_t b_dim_size = b_shape[b_corres_dim];
                size_t b_idx = (b_dim_size > 1) ? batch_coord : 0;
                b_batch_offset += b_idx * b_strides[b_corres_dim];
            } 
        }
                    
                    
        T sum{};
        for (size_t k = 0; k < n; ++k) {
            size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
            size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
            sum += A[a_idx] * B[b_idx];
        }
        
        size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
        output[out_idx] = sum;
    }

    // Specializations for bfloat16 and half (similar structure)
    __global__ void batched_matmul_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* output,
                                    const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                    size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                    size_t total_batches)
    {
        size_t batch_idx = blockIdx.z;
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        if (i >= m || j >= p) return;

        // Calculate batch offsets
        size_t a_batch_offset = 0;
        size_t b_batch_offset = 0;
        size_t out_batch_offset = 0;

        // FIXED: Proper batch offset calculation
        size_t temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim) {
            size_t batch_dim_size = out_shape[dim];
            size_t batch_coord = temp_batch % batch_dim_size;
            temp_batch /= batch_dim_size;
            
            // Calculate offsets using the actual batch coordinates
            if (dim < a_ndim - 2) {
                a_batch_offset += batch_coord * a_strides[dim];
            }
            if (dim < b_ndim - 2) {
                b_batch_offset += batch_coord * b_strides[dim];
            }
            out_batch_offset += batch_coord * out_strides[dim];
        }

        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
            size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
            sum += __bfloat162float(A[a_idx]) * __bfloat162float(B[b_idx]);
        }
        
        size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
        output[out_idx] = __float2bfloat16(sum);
    }

    __global__ void batched_matmul_kernel(const __half* A, const __half* B, __half* output,
                                        const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                        const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                        size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                        size_t total_batches)
    {
        size_t batch_idx = blockIdx.z;
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        if (i >= m || j >= p) return;

        // Calculate batch offsets
    size_t a_batch_offset = 0;
    size_t b_batch_offset = 0;
    size_t out_batch_offset = 0;

    // FIXED: Proper batch offset calculation
    size_t temp_batch = batch_idx;
    for (int dim = out_ndim - 3; dim >= 0; --dim) {
        size_t batch_dim_size = out_shape[dim];
        size_t batch_coord = temp_batch % batch_dim_size;
        temp_batch /= batch_dim_size;
        
        // Calculate offsets using the actual batch coordinates
        if (dim < a_ndim - 2) {
            a_batch_offset += batch_coord * a_strides[dim];
        }
        if (dim < b_ndim - 2) {
            b_batch_offset += batch_coord * b_strides[dim];
        }
        out_batch_offset += batch_coord * out_strides[dim];
    }

        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
            size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
            sum += __half2float(A[a_idx]) * __half2float(B[b_idx]);
        }
        
        size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
        output[out_idx] = __float2half(sum);
    }

    void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream) //✨✨✨
    {
        const auto& a_shape = A.shape().dims;
        const auto& b_shape = B.shape().dims;
        const auto& out_shape = output.shape().dims;
        
        size_t a_ndim = a_shape.size();
        size_t b_ndim = b_shape.size();
        size_t out_ndim = out_shape.size();
        
        // Calculate total batches
        size_t total_batches = 1;
        for (int i = 0; i < out_ndim - 2; ++i) {
            total_batches *= out_shape[i];
        }

        // Matrix dimensions
        size_t m = a_shape[a_ndim - 2];
        size_t p = b_shape[b_ndim - 1];

        // 3D grid for batches
        dim3 block(16, 16);
        dim3 grid((p + block.x - 1) / block.x, 
                  (m + block.y - 1) / block.y, 
                  total_batches);

        // Device memory allocation for shapes and strides
        size_t *d_a_shape, *d_b_shape, *d_out_shape;
        size_t *d_a_strides, *d_b_strides, *d_out_strides;

        cudaMalloc(&d_a_shape, a_ndim * sizeof(size_t));
        cudaMalloc(&d_b_shape, b_ndim * sizeof(size_t));
        cudaMalloc(&d_out_shape, out_ndim * sizeof(size_t));
        cudaMalloc(&d_a_strides, a_ndim * sizeof(size_t));
        cudaMalloc(&d_b_strides, b_ndim * sizeof(size_t));
        cudaMalloc(&d_out_strides, out_ndim * sizeof(size_t));

        // Copy data to device
        cudaMemcpyAsync(d_a_shape, a_shape.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_b_shape, b_shape.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_out_shape, out_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_a_strides, A.stride().strides.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_b_strides, B.stride().strides.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_out_strides, output.stride().strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨

        dispatch_by_dtype(A.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            const T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            T* out_ptr = output.data<T>();

            batched_matmul_kernel<<<grid, block, 0, stream>>>( //✨✨✨
                a_ptr, b_ptr, out_ptr,
                d_a_shape, d_b_shape, d_out_shape,
                d_a_strides, d_b_strides, d_out_strides,
                a_ndim, b_ndim, out_ndim, total_batches
            );
            
            //✨✨✨
            // cudaError_t err = cudaGetLastError();
            // if (err != cudaSuccess)
            // {
            //     // Free device memory before throwing
            //     cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_out_shape);
            //     cudaFree(d_a_strides); cudaFree(d_b_strides); cudaFree(d_out_strides);
            //     throw std::runtime_error("Batched Matmul Cuda Kernel Failed: " + 
            //     std::string(cudaGetErrorString(err)));
            // }

            // err = cudaDeviceSynchronize();
            // if (err != cudaSuccess) {
            //     cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_out_shape);
            //     cudaFree(d_a_strides); cudaFree(d_b_strides); cudaFree(d_out_strides);
            //     throw std::runtime_error("Batched Matmul Cuda Kernel Sync Failed: " + 
            //         std::string(cudaGetErrorString(err)));
            // }
        });

        // Free device memory
        cudaFree(d_a_shape);
        cudaFree(d_b_shape);
        cudaFree(d_out_shape);
        cudaFree(d_a_strides);
        cudaFree(d_b_strides);
        cudaFree(d_out_strides);
    }
}
#endif