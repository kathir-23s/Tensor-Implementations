// include/utilities/ReductionKernels.cuh - FIXED VERSION
#pragma once

#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

// ✅ Use native CUDA types directly
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include "ReductionOps.h"
#include <limits>

namespace OwnTensor {
namespace cuda {

// Type aliases for consistency with CPU API
using float16_t = __half;
using bfloat16_t = __nv_bfloat16;


// ═══════════════════════════════════════════════════════════
// HELPER TYPE TRAITS
// ═══════════════════════════════════════════════════════════

template <typename T>
constexpr bool is_reduced_precision_v = 
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

template <typename T>
constexpr bool is_any_float_v = 
    std::is_floating_point_v<T> || is_reduced_precision_v<T>;

// ═══════════════════════════════════════════════════════════
// GPU TYPE CONVERSION HELPERS (using native intrinsics)
// ═══════════════════════════════════════════════════════════

template<typename T>
__device__ __host__ inline float load_and_convert(const T* ptr, int64_t idx) {
    if constexpr (std::is_same_v<T, float16_t>) {
        return __half2float(ptr[idx]);  // ✅ Native conversion
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return __bfloat162float(ptr[idx]);  // ✅ Native conversion
    } else {
        return static_cast<float>(ptr[idx]);
    }
}

template<typename T>
__device__ __host__ inline void convert_and_store(T* ptr, int64_t idx, float val) {
    if constexpr (std::is_same_v<T, float16_t>) {
        ptr[idx] = __float2half(val);  // ✅ Native conversion
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ptr[idx] = __float2bfloat16(val);  // ✅ Native conversion
    } else {
        ptr[idx] = static_cast<T>(val);
    }
}

// =================================================================
// KERNEL DECLARATIONS - FIXED OUTPUT TYPES
// =================================================================

// Main value reduction kernel - NOW USES OutputT for correct type handling
template<typename T, typename OutputT, template<typename> class OpType>
__global__ void reduce_kernel(
    const T* __restrict__ input_data,
    OutputT* __restrict__ output_data,  // ✅ FIXED: Use actual output type
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved);

// Index reduction kernel (argmin/argmax)
template<typename T, template<typename> class OpType>
__global__ void reduce_index_kernel(
    const T* __restrict__ input_data,
    int64_t* __restrict__ output_data,
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved);

// Mean reduction kernel (high precision)
template<typename T, template<typename> class SumOpType>
__global__ void reduce_mean_kernel(
    const T* __restrict__ input_data,
    T* __restrict__ output_data,
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved);

} // namespace cuda
} // namespace OwnTensor

#endif // REDUCTION_KERNELS_CUH