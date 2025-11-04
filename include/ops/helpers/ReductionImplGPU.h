#pragma once

#ifndef OWNTENSOR_REDUCTIONS_IMPL_GPU_H
#define OWNTENSOR_REDUCTIONS_IMPL_GPU_H

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ReductionUtils.h"
#include "ReductionOps.h"
#include <vector>
#include <cstdint>
#include <driver_types.h>//✨✨✨
#ifdef WITH_CUDA
#include <cuda_runtime.h> // <-- CORRECT: Outside namespace
#endif//✨✨✨
namespace OwnTensor {
namespace detail {

#ifdef WITH_CUDA

// Forward declarations only (no implementation here!)
template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input, 
                               const std::vector<int64_t>& normalized_axes, 
                               bool keepdim, cudaStream_t stream);//✨✨✨

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input, 
                                     const std::vector<int64_t>& normalized_axes, 
                                     bool keepdim, cudaStream_t stream);//✨✨✨

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor& input, 
                         const std::vector<int64_t>& normalized_axes, 
                         bool keepdim, cudaStream_t stream);//✨✨✨

#endif // WITH_CUDA

} // namespace detail
} // namespace OwnTensor

#endif // OWNTENSOR_REDUCTIONS_IMPL_GPU_H