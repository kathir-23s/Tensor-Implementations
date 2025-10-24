#include <cuda_runtime.h>
#include <cstdint>

namespace OwnTensor {

template<int MaxDims>
__device__ __forceinline__
void linear_to_index(int64_t linear, const int64_t* dims, int32_t ndim, int64_t* out_idx) {
    // Row-major: last dimension changes fastest
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t dim = dims[d];
        out_idx[d] = linear % dim;
        linear /= dim;
    }
}

template<int MaxDims>
__global__ void contiguous_strided_copy_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t total_elems,
    const int64_t* __restrict__ dims,
    const int64_t* __restrict__ strides,
    int32_t ndim,
    int64_t storage_offset_elems,  // Should be 0 when src is from data()
    int32_t elem_size)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;

    // Convert output linear index to multi-index
    int64_t idx[MaxDims];
    linear_to_index<MaxDims>(i, dims, ndim, idx);

    // Compute source element offset using ONLY strides
    // storage_offset_elems should be 0 since src pointer is already offset
    int64_t elem_off = storage_offset_elems;  // This should be 0!
    
    #pragma unroll
    for (int d = 0; d < ndim; ++d) {
        elem_off += idx[d] * strides[d];
    }

    // Compute pointers
    const uint8_t* src_ptr = src + elem_off * elem_size;
    uint8_t* dst_ptr = dst + i * elem_size;

    // Copy element bytes
    for (int k = 0; k < elem_size; ++k) {
        dst_ptr[k] = src_ptr[k];
    }
}

extern "C" void contiguous_strided_copy_cuda(
    const void* src, void* dst,
    int64_t total_elems,
    const int64_t* dims, const int64_t* strides, int32_t ndim,
    int64_t storage_offset, int32_t elem_size,
    cudaStream_t stream)
{
    constexpr int MaxDims = 12;
    constexpr int Threads = 256;
    int64_t blocks = (total_elems + Threads - 1) / Threads;

    contiguous_strided_copy_kernel<MaxDims><<<blocks, Threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src),
        reinterpret_cast<uint8_t*>(dst),
        total_elems,
        dims,
        strides,
        ndim,
        storage_offset,  // Should be 0!
        elem_size
    );
    
    // IMPORTANT: Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error already happened, but let's not crash here
    }
}

} // namespace OwnTensor