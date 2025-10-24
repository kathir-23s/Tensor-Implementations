#pragma once

#ifndef OWNTENSOR_REDUCTIONS_H
#define OWNTENSOR_REDUCTIONS_H

#include "core/Tensor.h" // Defines the OwnTensor::Tensor class and related structs
#include <vector>
#include <cstdint> // For int64_t

// CRITICAL STEP: Include the implementation header which contains ALL template definitions.
// This allows the non-template functions in reductions.cpp to instantiate the templates.
#include "ops/helpers/ReductionImpl.h" 

namespace OwnTensor { // <<< START OF THE PUBLIC API NAMESPACE

// NOTE: Using default arguments to eliminate overload ambiguity.

// =================================================================
// 1. Core Reductions
// =================================================================
Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_product(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);

// =================================================================
// 2. NaN-Aware Reductions
// =================================================================
Tensor reduce_nansum(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_nanproduct(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_nanmin(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_nanmax(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_nanmean(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);

// =================================================================
// 3. Index Reductions
// =================================================================
// Note: Index reductions return a Tensor with Dtype::Int64
Tensor reduce_argmin(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_argmax(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);

// =================================================================
// 4. NaN-Aware Index Reductions
// =================================================================
Tensor reduce_nanargmin(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);
Tensor reduce_nanargmax(const Tensor& input, const std::vector<int64_t>& axes = {}, bool keepdim = false);

} // namespace OwnTensor

#endif // OWNTENSOR_REDUCTIONS_H
