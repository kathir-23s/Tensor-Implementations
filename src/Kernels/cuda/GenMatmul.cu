// #ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

#include "ops/Matmul.cuh"
#include "core/Tensor.h"

#define TILE_WIDTH 32 // for now let the tile size be (16 x 16)

namespace OwnTensor {
    /*
    Implementation of tiled MatMul to address the memory bound
    limitation of the current batched_matmul_kernel.
    The width param in the book should be replaced by 'k',
    the number of cols in the first matrix and rows in the second matrix. 
    */
    template<typename T>
    __global__
    void tiled_matmul_kernel(const T* A, const T* B, T* output, 
                            const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                            const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                            size_t a_ndim, size_t b_ndim, size_t out_ndim,
                            int m, int n, int k,
                            size_t total_batches) {
        /*
        allocating space for a tile or portion of the matrix in shared memory
        to reduce the frequency and latency of global memory access.
        */
        __shared__ T s_A[TILE_WIDTH][TILE_WIDTH + 1]; // to avoid bank conflicts
        __shared__ T s_B[TILE_WIDTH][TILE_WIDTH + 1]; // padding technique

        /*
        changed from size_t(8 bytes) to int(4 bytes) to reduce the memory on the register.
        changed the variable names for better context.
        batch_idx, rows and cols will be stored in the register since they are scalar values.
        */
        int batch_idx = blockIdx.z; // largest number int =>  2,147,483,647.
        int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        if (batch_idx >= total_batches) return;
        if (row >= m || col >= n) return;

        /*
        don't know what this, reduced data type to int to reduce the pressure on register
        */
        // Calculate batch offsets
        int a_batch_offset = 0;
        int b_batch_offset = 0;
        int out_batch_offset = 0;
                    
        // FIXED: Proper batch offset calculation
        int temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim){
            int batch_dim_size = out_shape[dim];
            int batch_coord = temp_batch % batch_dim_size;
            temp_batch /= batch_dim_size;
            out_batch_offset += batch_coord * out_strides[dim];
            // RIGHT-ALIGNED: Calculating corresponding dimensions for A and B
            int a_corres_dim = dim - (out_ndim - 2 - (a_ndim - 2));
            int b_corres_dim = dim - (out_ndim - 2 - (b_ndim - 2));
            // For A and B: Right aligned broadcasting rules
            if (dim >= out_ndim - 2 - (a_ndim - 2))
            {
                int a_dim_size = a_shape[a_corres_dim];
                int a_idx = (a_dim_size > 1) ? batch_coord : 0;
                a_batch_offset += a_idx * a_strides[a_corres_dim];
            }
            if (dim >= out_ndim - 2 - (b_ndim - 2))
            {
                int b_dim_size = b_shape[b_corres_dim];
                int b_idx = (b_dim_size > 1) ? batch_coord : 0;
                b_batch_offset += b_idx * b_strides[b_corres_dim];
            } 
        }

        T Psum = 0; // partial sum (Psum is stored in a Register)

        // Loop over the reduction dimension K in phases (ph)
        for(int ph = 0; ph < ceil((float)k/TILE_WIDTH); ++ph){ 
            // Incorporate batch offset and strides for accurate access.
            // A[M, K] access: (A Row index) * A_stride[M] + (A Col index) * A_stride[K]
            int a_global_idx = a_batch_offset + 
                               row * a_strides[a_ndim - 2] + // Row index (M dim)
                               (ph * TILE_WIDTH + threadIdx.x) * a_strides[a_ndim - 1]; // Col index (K dim)

            // B[K, N] access: (B Row index) * B_stride[K] + (B Col index) * B_stride[N]
            int b_global_idx = b_batch_offset + 
                               (ph * TILE_WIDTH + threadIdx.y) * b_strides[b_ndim - 2] + // Row index (K dim)
                               col * b_strides[b_ndim - 1]; // Col index (N dim)

            if((row < m) && ((ph * TILE_WIDTH + threadIdx.x) < k)){
                s_A[threadIdx.y][threadIdx.x] = A[a_global_idx];
            } else {
                s_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
            if(((ph * TILE_WIDTH + threadIdx.y) < k) && (col < n)){
                s_B[threadIdx.y][threadIdx.x] = B[b_global_idx];
            } else {
                s_B[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads(); // Wait for all threads to finish loading the tile

            for(int i = 0; i < TILE_WIDTH; ++i){
                // Read from shared memory [6]
                Psum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            }
            __syncthreads(); // Wait for all threads to finish computation phase 
        }
        // Final output write
        int out_idx = out_batch_offset + row * out_strides[out_ndim - 2] + col * out_strides[out_ndim - 1];
        if((row < m) && (col < n)){
            output[out_idx] = Psum;
        }
    }

    template <typename T>
    __global__ void batched_matmul_kernel(const T* A, const T* B, T* output,
                                        const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                        const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                        size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                        int m, int n, int k,
                                        size_t total_batches)
    {
        /*
        changed from size_t(8 bytes) to int(4 bytes) to reduce the memory on the register.
        changed the variable names for better context.
        batch_idx, rows and cols will be stored in the register since they are scalar values.
        I think that we are not considering cases where the number of dims exceeds 3. 
        */
        int batch_idx = blockIdx.z; // largest number int =>  2,147,483,647.
        int rows = blockIdx.y * blockDim.y + threadIdx.y;
        int cols = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;
        if (rows >= m || cols >= n) return;

        /*
        don't know what this, reduced data type to int to reduce the pressure on register
        */
        // Calculate batch offsets
        int a_batch_offset = 0;
        int b_batch_offset = 0;
        int out_batch_offset = 0;
                    
        // FIXED: Proper batch offset calculation
        int temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim){
            int batch_dim_size = out_shape[dim];
            int batch_coord = temp_batch % batch_dim_size;
            temp_batch /= batch_dim_size;
            out_batch_offset += batch_coord * out_strides[dim];
            // RIGHT-ALIGNED: Calculating corresponding dimensions for A and B
            int a_corres_dim = dim - (out_ndim - 2 - (a_ndim - 2));
            int b_corres_dim = dim - (out_ndim - 2 - (b_ndim - 2));
            // For A and B: Right aligned broadcasting rules
            if (dim >= out_ndim - 2 - (a_ndim - 2))
            {
                int a_dim_size = a_shape[a_corres_dim];
                int a_idx = (a_dim_size > 1) ? batch_coord : 0;
                a_batch_offset += a_idx * a_strides[a_corres_dim];
            }
            if (dim >= out_ndim - 2 - (b_ndim - 2))
            {
                int b_dim_size = b_shape[b_corres_dim];
                int b_idx = (b_dim_size > 1) ? batch_coord : 0;
                b_batch_offset += b_idx * b_strides[b_corres_dim];
            } 
        }
                    
                    
        T sum{};
        for (int i = 0; i < k; ++i) {
            int a_idx = a_batch_offset + rows * a_strides[a_ndim - 2] + i * a_strides[a_ndim - 1];
            int b_idx = b_batch_offset + i * b_strides[b_ndim - 2] + cols * b_strides[b_ndim - 1];
            sum += A[a_idx] * B[b_idx]; // => here the highest latency occurs. (L1/TEX stalls)
        }
        
        int out_idx = out_batch_offset + rows * out_strides[out_ndim - 2] + cols * out_strides[out_ndim - 1];
        output[out_idx] = sum;
    }

    // Specializations for bfloat16 and half (similar structure)
    __global__ void batched_matmul_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* output,
                                    const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                    size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                    int m, int n, int k,
                                    size_t total_batches)
    {
        int batch_idx = blockIdx.z;
        int rows = blockIdx.y * blockDim.y + threadIdx.y;
        int cols = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        // size_t m = a_shape[a_ndim - 2];
        // size_t n = a_shape[a_ndim - 1];
        // size_t p = b_shape[b_ndim - 1];

        if (rows >= m || cols >= n) return;

        // Calculate batch offsets
        int a_batch_offset = 0;
        int b_batch_offset = 0;
        int out_batch_offset = 0;

        // FIXED: Proper batch offset calculation
        int temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim) {
            int batch_dim_size = out_shape[dim];
            int batch_coord = temp_batch % batch_dim_size;
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
        for (int i = 0; i < k; ++i) {
            int a_idx = a_batch_offset + rows * a_strides[a_ndim - 2] + i * a_strides[a_ndim - 1];
            int b_idx = b_batch_offset + i * b_strides[b_ndim - 2] + cols * b_strides[b_ndim - 1];
            sum += __bfloat162float(A[a_idx]) * __bfloat162float(B[b_idx]);
        }
        
        int out_idx = out_batch_offset + rows * out_strides[out_ndim - 2] + cols * out_strides[out_ndim - 1];
        output[out_idx] = __float2bfloat16(sum);
    }

    __global__ void batched_matmul_kernel(const __half* A, const __half* B, __half* output,
                                        const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                        const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                        size_t a_ndim, size_t b_ndim, size_t out_ndim, 
                                        int m, int n, int k,
                                        size_t total_batches)
    {
        int batch_idx = blockIdx.z;
        int rows = blockIdx.y * blockDim.y + threadIdx.y;
        int cols = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        if (rows >= m || cols >= n) return;

        // Calculate batch offsets
        int a_batch_offset = 0;
        int b_batch_offset = 0;
        int out_batch_offset = 0;

        // FIXED: Proper batch offset calculation
        int temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim) {
            int batch_dim_size = out_shape[dim];
            int batch_coord = temp_batch % batch_dim_size;
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
        for (int i = 0; i < k; ++i) {
            int a_idx = a_batch_offset + rows * a_strides[a_ndim - 2] + i * a_strides[a_ndim - 1];
            int b_idx = b_batch_offset + i * b_strides[b_ndim - 2] + cols * b_strides[b_ndim - 1];
            sum += __half2float(A[a_idx]) * __half2float(B[b_idx]);
        }
        
        int out_idx = out_batch_offset + rows * out_strides[out_ndim - 2] + cols * out_strides[out_ndim - 1];
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

        /*
        we could have m,n,k to be consistent with the general matmul signature,
        where m is the number of rows of the first matrix,
        n is the number of cols of the second matrix,
        and k is the number of cols in the first matrix and number of rows in the second matrix
        */
        // Matrix dimensions
        int m = a_shape[a_ndim - 2]; // number of rows of the first matrix
        int n = b_shape[b_ndim - 1]; // number of cols of the second matrix
        int k = a_shape[a_ndim - 1]; // number of cols and rows in both matrix

        // 3D grid for batches
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x,  // =>> here
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

            tiled_matmul_kernel<<<grid, block, 0, stream>>>( //✨✨✨
                a_ptr, b_ptr, out_ptr,
                d_a_shape, d_b_shape, d_out_shape,
                d_a_strides, d_b_strides, d_out_strides,
                a_ndim, b_ndim, out_ndim, m, n, k, total_batches
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
// #endif