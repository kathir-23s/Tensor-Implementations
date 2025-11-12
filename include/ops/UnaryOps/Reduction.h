#pragma once

#ifndef OWNTENSOR_REDUCTIONS_H
#define OWNTENSOR_REDUCTIONS_H

#include "core/Tensor.h" // Defines the OwnTensor::Tensor class and related structs
#include <vector>
#include <cstdint> // For int64_t
#ifdef WITH_CUDA//✨✨✨
#include <driver_types.h>
#endif//✨✨✨
// CRITICAL STEP: Include the implementation header which contains ALL template definitions.
// This allows the non-template functions in reductions.cpp to instantiate the templates.
#include "ops/helpers/ReductionImpl.h" 

namespace OwnTensor { // <<< START OF THE PUBLIC API NAMESPACE

// NOTE: Using default arguments to eliminate overload ambiguity.

// =================================================================
// 1. Core Reductions
// =================================================================
Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_product(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨

// =================================================================
// 2. NaN-Aware Reductions
// =================================================================
Tensor reduce_nansum(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_nanproduct(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_nanmin(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_nanmax(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_nanmean(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨

// =================================================================
// 3. Index Reductions
// =================================================================
// Note: Index reductions return a Tensor with Dtype::Int64
Tensor reduce_argmin(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_argmax(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨

// =================================================================
// 4. NaN-Aware Index Reductions
// =================================================================
Tensor reduce_nanargmin(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨
Tensor reduce_nanargmax(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false, cudaStream_t stream = 0);//✨✨✨

// =================================================================
// 5. Boolean Reductions (Bool dtype only)
// =================================================================

/** Check if all elements (or along axes) are true
 * @param input Boolean tensor to reduce
 * @param axes Dimensions to reduce over (empty = all dimensions)
 * @param keepdim Keep reduced dimensions as size 1
 * @return Boolean tensor with True where all elements are true
 */
Tensor reduce_all(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                  bool keepdim = false, cudaStream_t stream = 0);

/** Check if any element (or along axes) is true
 * @param input Boolean tensor to reduce
 * @param axes Dimensions to reduce over (empty = all dimensions)
 * @param keepdim Keep reduced dimensions as size 1
 * @return Boolean tensor with True where any element is true
 */
Tensor reduce_any(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                  bool keepdim = false, cudaStream_t stream = 0);
// =================================================================
// VARIANCE & STANDARD DEVIATION OPERATIONS
// =================================================================

/** Compute variance over specified axes
 * @param correction Bessel's correction (default=1 for sample variance, 0 for population)
 * Formula: var = sum((x - mean)Â²) / (N - correction)
 */
Tensor reduce_var(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                      bool keepdim = false, int64_t correction = 1, cudaStream_t stream = 0);//✨✨✨

Tensor reduce_nanvar(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                          bool keepdim = false, int64_t correction = 1, cudaStream_t stream = 0);//✨✨✨

/** Compute standard deviation over specified axes
 * @param correction Bessel's correction (default=1 for sample std, 0 for population)
 * Formula: std = sqrt(var)
 */
Tensor reduce_std(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                  bool keepdim = false, int64_t correction = 1, cudaStream_t stream = 0);//✨✨✨   

Tensor reduce_nanstd(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                     bool keepdim = false, int64_t correction = 1, cudaStream_t stream = 0);//✨✨✨

//============================================================================================
// COMBINED STATISTICS (Efficient single-pass computation)
//============================================================================================

/** Returns tuple: (variance, mean) - More efficient than separate calls */
std::pair<Tensor, Tensor> reduce_var_mean(const Tensor& input, 
                                          const std::vector<int64_t>& axes = {}, 
                                          bool keepdim = false, 
                                          int64_t correction = 1, cudaStream_t stream = 0   );//✨✨✨

/** Returns tuple: (std, mean) - More efficient than separate calls */
std::pair<Tensor, Tensor> reduce_std_mean(const Tensor& input, 
                                          const std::vector<int64_t>& axes = {}, 
                                          bool keepdim = false, 
                                          int64_t correction = 1, cudaStream_t stream = 0   );//✨✨✨

//============================================================================================
// ORDER STATISTICS (Requires sorting - computationally expensive)
//============================================================================================

/** Compute median (50th percentile) along axes */
Tensor reduce_median(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                     bool keepdim = false, cudaStream_t stream = 0  );//✨✨✨

Tensor reduce_nanmedian(const Tensor& input, const std::vector<int64_t>& axes = {}, 
                        bool keepdim = false, cudaStream_t stream = 0  );//✨✨✨

// Note: Mode requires histogram-based approach - deferred to Phase 2
} // namespace OwnTensor


#endif // OWNTENSOR_REDUCTIONS_H
