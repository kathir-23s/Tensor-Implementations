// ============================================================================
// Create new file: include/core/ConversionKernels.cuh
// ============================================================================

#pragma once

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cstdint>

namespace OwnTensor {


// Forward declarations
template<typename T>
void convert_to_bool_cuda(const T* input, bool* output, int64_t n, cudaStream_t stream = 0);


} // namespace OwnTensor

#endif // WITH_CUDA