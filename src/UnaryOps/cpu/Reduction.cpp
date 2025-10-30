// src/UnaryOps/Reduction.cpp - FIXED: Use is_float() from DtypeTraits.h
#include "ops/UnaryOps/Reduction.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/ReductionUtils.h"
#include "ops/helpers/ReductionImpl.h"
#include "dtype/DtypeTraits.h"  // ← Provides is_float() and get_dtype_name()
#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <type_traits>

namespace OwnTensor {
using namespace detail;
namespace {

// // ✅ Base dispatcher for BASIC operations (works on all types)
// template <template <typename> class OpType>
// Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_reduction<T, OpType>(input, normalized_axes, keepdim);
//     });
// }

// ✅ Dispatcher for NaN-AWARE operations (only for floating point)
// Uses is_float() from DtypeTraits.h instead of custom function
// template <template <typename> class OpType>
// Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
//     // ✅ Use is_float() from DtypeTraits.h (teammate's function)
//     if (!is_float(input.dtype())) {
//         throw std::runtime_error(
//             "NaN-aware reductions are only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
//             "Got: " + get_dtype_name(input.dtype())
//         );
//     }
    
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_reduction<T, OpType>(input, normalized_axes, keepdim);
//     });
// }

// // ✅ Mean dispatcher for BASIC operations
// template <template <typename> class SumOpType>
// Tensor _reduce_mean_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_mean_kernel<T, SumOpType>(input, normalized_axes, keepdim);
//     });
// }

// ✅ Mean dispatcher for NaN-AWARE operations (only for floating point)
// template <template <typename> class SumOpType>
// Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
//     // ✅ Use is_float() from DtypeTraits.h
//     if (!is_float(input.dtype())) {
//         throw std::runtime_error(
//             "NaN-aware mean is only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
//             "Got: " + get_dtype_name(input.dtype())
//         );
//     }
    
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_mean_kernel<T, SumOpType>(input, normalized_axes, keepdim);
//     });
// }

} // anonymous namespace

// =================================================================
// 1. Core Reductions (All types supported)
// =================================================================
Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, SumOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_product(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ProductOp>(input, normalized_axes, keepdim);
    });
}
Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, MinOp>(input, normalized_axes, keepdim);
    });
}
Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, MaxOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_mean_kernel<T, SumOp>(input, normalized_axes, keepdim);
    });
}

// =================================================================
// 2. NaN-Aware Reductions (FLOATING POINT ONLY)
// =================================================================
Tensor reduce_nansum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanSumOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_nanproduct(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanProductOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_nanmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanMinOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_nanmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanMaxOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_nanmean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, keepdim);
    });
}

// =================================================================
// 3. Index Reductions (All types supported)
// =================================================================
Tensor reduce_argmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ArgMinOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_argmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ArgMaxOp>(input, normalized_axes, keepdim);
    });
}

// =================================================================
// 4. NaN-Aware Index Reductions (FLOATING POINT ONLY)
// =================================================================
Tensor reduce_nanargmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanArgMinOp>(input, normalized_axes, keepdim);
    });
}

Tensor reduce_nanargmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanArgMaxOp>(input, normalized_axes, keepdim);
    });
}

} // namespace OwnTensor