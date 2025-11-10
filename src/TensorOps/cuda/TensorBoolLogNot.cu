#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"

namespace OwnTensor
{   
template<typename T>
__global__ void bool_not_kernel(const T* a, bool* output, size_t n)//✨✨✨  
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Placeholder operation: set output to true if a[idx] equals b[idx], else false
        output[idx] = !(a[idx]);
    }
}


//cuda_bool_eq_outplace
void cuda_logical_not_outplace(const Tensor &A, Tensor &output, cudaStream_t stream)
    {
        
        size_t total_elems = output.numel();
        size_t block_size = 256;
        size_t grid_size = (total_elems + block_size - 1) / block_size;

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
                          {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        
        bool* output_ptr = output.data<bool>();
        
        bool_not_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr,output_ptr, total_elems);
         });
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
}
#endif // WITH_CUDA

