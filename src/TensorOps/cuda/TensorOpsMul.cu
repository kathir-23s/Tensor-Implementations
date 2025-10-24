#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"

namespace OwnTensor
{

template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = a[idx] * b[idx];
    }
}

template <>
__global__ void mul_kernel<__half>(const __half* a, const __half* b, __half* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __hmul(a[idx], b[idx]);
    }
}

template <>
__global__ void mul_kernel<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = __hmul(a[idx],b[idx]);
    }
}

void cuda_mul_tensor(const Tensor& A, const Tensor& B, Tensor& output)
{
    size_t total_elems = A.numel();
    size_t block_size = 256;
    size_t grid_size = (total_elems + block_size - 1) / block_size;
    std::cout << "Multiplication CUDA\n";

    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        const T* b_ptr = B.data<T>();
        T* output_ptr = output.data<T>();

        mul_kernel<<<grid_size, block_size>>>(a_ptr, b_ptr, output_ptr, total_elems);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Multiplication CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Multiplication CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
        }
    });
}

/*########################################################
            TENSOR INPLACE CUDA KERNELS
##########################################################*/


template <typename T>
    __global__ void mul_inplace_kernel(T* lhs, const T* rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] *= rhs[idx];
        }
    }

    template <>
    __global__ void mul_inplace_kernel<__half>(__half* lhs, const __half* rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] = __hmul(lhs[idx], rhs[idx]);
        }
    }

    template <>
    __global__ void mul_inplace_kernel<__nv_bfloat16>(__nv_bfloat16* lhs, const __nv_bfloat16* rhs, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            lhs[idx] = __hmul(lhs[idx],rhs[idx]);
        }
    }

    void cuda_mul_tensor_inplace(Tensor& A, const Tensor& B)
        {
            size_t total_elems = A.numel();
            size_t block_size = 256;
            size_t grid_size = (total_elems + block_size - 1) / block_size;

            std::cout << "Addition Inplace CUDA\n";

            dispatch_by_dtype(A.dtype(), [&](auto dummy)
            {
                using T = decltype(dummy);
                T* a_ptr = A.data<T>();
                const T* b_ptr = B.data<T>();
                 
                mul_inplace_kernel<<<grid_size, block_size>>>(a_ptr, b_ptr, total_elems);
                
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error("Addition CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
                }
                
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    throw std::runtime_error("Addition CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
                }
            });
        }

}

#endif
