#include "Reduction.h"
#include "TensorDispatch.h"   // Provides dispatch_by_dtype (now in OwnTensor:: scope)
#include "ReductionUtils.h"   // Provides normalize_axes, calculate_reduced_count (now in OwnTensor::detail:: scope)
#include "ReductionImpl.h"   // Provides ALL Op structs and ALL dispatch_reduction implementations (now in OwnTensor::detail:: scope)

#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <type_traits> // Required for std::is_integral_v

namespace OwnTensor {

namespace { // <<< Anonymous namespace for private helpers within OwnTensor

/**
 * @brief Base function to handle boilerplate checks (device, axes normalization, etc.)
 * and dispatch the core reduction logic.
 * @tparam OpType The base Operation Functor struct (e.g., detail::SumOp).
 * @param input The input Tensor.
 * @param axes The user-provided axes.
 * @param keepdim Whether to keep the reduced dimensions as size 1.
 * @return The result Tensor.
 */
template <template <typename> class OpType> // Template template parameter for the base Op
Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
   
    
    // 1. Normalize axes using the function from the detail namespace
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    // 2. Dispatch based on Dtype using the general dispatch utility
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        // T_val is an empty instance of the C++ type (e.g., float{}, int32_t{})
        using T = decltype(T_val);
        // Call the internal reduction kernel dispatcher
        return detail::dispatch_reduction<T, OpType>(input, normalized_axes, keepdim);
    });
}

/**
 * @brief Base function to handle boilerplate checks and dispatch the mean reduction logic.
 * @tparam SumOpType The base Sum Operation Functor struct (e.g., detail::SumOp or detail::NanSumOp).
 * @param input The input Tensor.
 * @param axes The user-provided axes.
 * @param keepdim Whether to keep the reduced dimensions as size 1.
 * @return The result Tensor.
 */
template <template <typename> class SumOpType>
Tensor _reduce_mean_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    

    // 1. Normalize axes
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    // 2. Dispatch based on Dtype
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        // Call the internal mean kernel dispatcher
        return detail::dispatch_mean_kernel<T, SumOpType>(input, normalized_axes, keepdim);
    });
}

} // namespace (end of anonymous namespace)

// =================================================================
// 1. Core Reductions (Public Definitions in OwnTensor)
// =================================================================
Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::SumOp>(input, axes, keepdim);
}

Tensor reduce_product(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::ProductOp>(input, axes, keepdim);
}

Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::MinOp>(input, axes, keepdim);
}

Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::MaxOp>(input, axes, keepdim);
}

Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    // Mean is based on the SumOp
    return _reduce_mean_dispatcher<detail::SumOp>(input, axes, keepdim);
}


// =================================================================
// 2. NaN-Aware Reductions
// =================================================================
Tensor reduce_nansum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::NanSumOp>(input, axes, keepdim);
}

Tensor reduce_nanproduct(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::NanProductOp>(input, axes, keepdim);
}

Tensor reduce_nanmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::NanMinOp>(input, axes, keepdim);
}

Tensor reduce_nanmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::NanMaxOp>(input, axes, keepdim);
}

Tensor reduce_nanmean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    // NanMean is based on the NanSumOp
    return _reduce_mean_dispatcher<detail::NanSumOp>(input, axes, keepdim);
}


// =================================================================
// 3. Index Reductions
// =================================================================
Tensor reduce_argmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::ArgMinOp>(input, axes, keepdim);
}

Tensor reduce_argmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::ArgMaxOp>(input, axes, keepdim);
}

// =================================================================
// 4. NaN-Aware Index Reductions
// =================================================================
Tensor reduce_nanargmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::NanArgMinOp>(input, axes, keepdim);
}

Tensor reduce_nanargmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    return _reduce_dispatcher<detail::NanArgMaxOp>(input, axes, keepdim);
}

} // namespace OwnTensor
