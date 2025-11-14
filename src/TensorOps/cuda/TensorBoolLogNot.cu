#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"

namespace OwnTensor
{   
// ✅ FIXED: Convert to bool inline, then apply logical AND
template<typename T>
__global__ void bool_not_kernel(const T* a,  bool* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // ✅ Convert to bool: 0 → false, non-zero → true
        bool a_bool = (a[idx] != T(0));
        //bool b_bool = (b[idx] != T(0));
        output[idx] = !a_bool;
    }
}

// ✅ Specialization for __half (needs special comparison)
template<>
__global__ void bool_not_kernel<__half>(const __half* a, bool* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __half zero = __float2half(0.0f);
        bool a_bool = !__heq(a[idx], zero);  // true if not equal to zero
        //bool b_bool = !__heq(b[idx], zero);
        output[idx] = !a_bool ;
    }
}

// ✅ Specialization for __nv_bfloat16
template<>
__global__ void bool_not_kernel<__nv_bfloat16>(const __nv_bfloat16* a,  bool* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        bool a_bool = !__heq(a[idx], zero);
        //bool b_bool = !__heq(b[idx], zero);
        output[idx] = !a_bool;
    }
}

// template<typename T>
// __global__ void bool_and_kernel_broadcast(const T* a, const T* b, bool* output,
//                                       const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
//                                       const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
//                                       size_t a_ndim, size_t b_ndim, size_t out_ndim,
//                                       size_t total_elems)
// {
//     size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (linear_idx >= total_elems) return;

//     size_t a_bcast_strides[8];
//     size_t b_bcast_strides[8];
    
//     for (size_t i = 0; i < out_ndim; ++i) {
//         size_t a_dim_idx = a_ndim - out_ndim + i;
//         size_t b_dim_idx = b_ndim - out_ndim + i;
        
//         if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
//             a_bcast_strides[i] = a_strides[a_dim_idx];
//         }
//         if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
//             b_bcast_strides[i] = b_strides[b_dim_idx];
//         }
//     }
    
//     size_t coords[8];
//     size_t temp_idx = linear_idx;
//     for (int dim = out_ndim - 1; dim >= 0; --dim) {
//         coords[dim] = temp_idx % out_shape[dim];
//         temp_idx /= out_shape[dim];
//     }
    
//     size_t a_idx = 0;
//     size_t b_idx = 0;
//     for (size_t dim = 0; dim < out_ndim; ++dim) {
//         a_idx += coords[dim] * a_bcast_strides[dim];
//         b_idx += coords[dim] * b_bcast_strides[dim];
//     }
    
//     // ✅ Convert to bool then apply logical AND
//     bool a_bool = (a[a_idx] != T(0));
//     bool b_bool = (b[b_idx] != T(0));
//     output[linear_idx] = a_bool && b_bool;
// }

// // ✅ Specializations for broadcast kernels
// template<>
// __global__ void bool_and_kernel_broadcast<__half>(const __half* a, const __half* b, bool* output,
//                                       const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
//                                       const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
//                                       size_t a_ndim, size_t b_ndim, size_t out_ndim,
//                                       size_t total_elems)
// {
//     size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (linear_idx >= total_elems) return;

//     size_t a_bcast_strides[8] = {0};
//     //size_t b_bcast_strides[8] = {0};
    
//     for (size_t i = 0; i < out_ndim; ++i) {
//         size_t a_dim_idx = a_ndim - out_ndim + i;
//         //size_t b_dim_idx = b_ndim - out_ndim + i;
        
//         if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
//             a_bcast_strides[i] = a_strides[a_dim_idx];
//         }
//         // if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
//         //     b_bcast_strides[i] = b_strides[b_dim_idx];
//         // }
//     }
    
//     size_t coords[8];
//     size_t temp_idx = linear_idx;
//     for (int dim = out_ndim - 1; dim >= 0; --dim) {
//         coords[dim] = temp_idx % out_shape[dim];
//         temp_idx /= out_shape[dim];
//     }
    
//     size_t a_idx = 0;
//     //size_t b_idx = 0;
//     for (size_t dim = 0; dim < out_ndim; ++dim) {
//         a_idx += coords[dim] * a_bcast_strides[dim];
//         //b_idx += coords[dim] * b_bcast_strides[dim];
//     }
    
//     __half zero = __float2half(0.0f);
//     bool a_bool = !__heq(a[a_idx], zero);
//     //bool b_bool = !__heq(b[b_idx], zero);
//     output[linear_idx] = !a_bool  ;
// }

// template<>
// __global__ void bool_not_kernel_broadcast<__nv_bfloat16>(const __nv_bfloat16* a,bool* output,
//                                       const size_t* a_shape, const size_t* out_shape,
//                                       const size_t* a_strides, const size_t* out_strides,
//                                       size_t a_ndim, size_t out_ndim,
//                                       size_t total_elems)
// {
//     size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (linear_idx >= total_elems) return;

//     size_t a_bcast_strides[8] = {0};
//     //size_t b_bcast_strides[8] = {0};
    
//     for (size_t i = 0; i < out_ndim; ++i) {
//         size_t a_dim_idx = a_ndim - out_ndim + i;
//         //size_t b_dim_idx = b_ndim - out_ndim + i;
        
//         if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
//             a_bcast_strides[i] = a_strides[a_dim_idx];
//         }
//     //     // if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
//     //     //     b_bcast_strides[i] = b_strides[b_dim_idx];
//     //     // }
//      }
    
//     size_t coords[8];
//     size_t temp_idx = linear_idx;
//     for (int dim = out_ndim - 1; dim >= 0; --dim) {
//         coords[dim] = temp_idx % out_shape[dim];
//         temp_idx /= out_shape[dim];
//     }
    
//     size_t a_idx = 0;
//    // size_t b_idx = 0;
//     for (size_t dim = 0; dim < out_ndim; ++dim) {
//         a_idx += coords[dim] * a_bcast_strides[dim];
//        // b_idx += coords[dim] * b_bcast_strides[dim];
//     }
    
//     __nv_bfloat16 zero = __float2bfloat16(0.0f);
//     bool a_bool = !__heq(a[a_idx], zero);
//     //bool b_bool = !__heq(b[b_idx], zero);
//     output[linear_idx] = !a_bool;
// }

// Host function remains the same
void cuda_logical_not_outplace(const Tensor &A,Tensor &output, cudaStream_t stream)
{
    // bool needs_broadcasting = (A.shape().dims != B.shape().dims);
    size_t total_elems = output.numel();
    size_t block_size = 256;
    size_t grid_size = (total_elems + block_size - 1) / block_size;

    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        //const T* b_ptr = B.data<T>();
        bool* output_ptr = output.data<bool>();
        
        // if (!needs_broadcasting) {
            bool_not_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr,  output_ptr, total_elems);
        // } else {
        //     const auto& a_shape = A.shape().dims;
        //     //const auto& b_shape = B.shape().dims;
        //     const auto& out_shape = output.shape().dims;
            
        //     size_t a_ndim = a_shape.size();
        //    // size_t b_ndim = b_shape.size();
        //     size_t out_ndim = out_shape.size();
            
        //     size_t *d_a_shape, *d_out_shape;
        //     size_t *d_a_strides, *d_out_strides;
            
        //     cudaMalloc(&d_a_shape, a_ndim * sizeof(size_t));
        //     //cudaMalloc(&d_b_shape, b_ndim * sizeof(size_t));
        //     cudaMalloc(&d_out_shape, out_ndim * sizeof(size_t));
        //     cudaMalloc(&d_a_strides, a_ndim * sizeof(size_t));
        //     //cudaMalloc(&d_b_strides, b_ndim * sizeof(size_t));
        //     cudaMalloc(&d_out_strides, out_ndim * sizeof(size_t));
            
        //     cudaMemcpyAsync(d_a_shape, a_shape.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        //    // cudaMemcpyAsync(d_b_shape, b_shape.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        //     cudaMemcpyAsync(d_out_shape, out_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        //     cudaMemcpyAsync(d_a_strides, A.stride().strides.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        //     //cudaMemcpyAsync(d_b_strides, B.stride().strides.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        //     cudaMemcpyAsync(d_out_strides, output.stride().strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            
        //     bool_not_kernel_broadcast<<<grid_size, block_size, 0, stream>>>(
        //         a_ptr,  output_ptr,
        //         d_a_shape, d_out_shape,
        //         d_a_strides,  d_out_strides,
        //         a_ndim,  out_ndim, total_elems
        //     );
            
        //     cudaFree(d_a_shape);
        //     //cudaFree(d_b_shape);
        //     cudaFree(d_out_shape);
        //     cudaFree(d_a_strides);
        //    // cudaFree(d_b_strides);
        //     cudaFree(d_out_strides);
        // }
    });
}

}
#endif // WITH_CUDA