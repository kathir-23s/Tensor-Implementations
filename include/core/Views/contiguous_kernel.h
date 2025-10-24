#pragma once
#include <cstdint>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {
#ifdef WITH_CUDA
extern "C" void contiguous_strided_copy_cuda(
    const void* src, void* dst,
    int64_t total_elems,
    const int64_t* dims, const int64_t* strides, int32_t ndim,
    int64_t storage_offset, int32_t elem_size,
    cudaStream_t stream);
#endif
} // namespace OwnTensor
