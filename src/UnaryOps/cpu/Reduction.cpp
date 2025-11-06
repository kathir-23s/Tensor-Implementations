// src/UnaryOps/Reduction.cpp - FIXED: Use is_float() from DtypeTraits.h
#include "ops/UnaryOps/Reduction.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/ReductionUtils.h"
#include "ops/helpers/ReductionImpl.h"
#include "dtype/DtypeTraits.h"  // ← Provides is_float() and get_dtype_name()
#include <driver_types.h>//✨✨✨
#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <type_traits>
#include "ops/UnaryOps/Arithmetics.h" // For sqrt()
namespace OwnTensor {
using namespace detail;
namespace {

// // ✅ Base dispatcher for BASIC operations (works on all types)
// template <template <typename> class OpType>
// Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_reduction<T, OpType>(input, normalized_axes, keepdim, stream);
//     });
// }

// ✅ Dispatcher for NaN-AWARE operations (only for floating point)
// Uses is_float() from DtypeTraits.h instead of custom function
// template <template <typename> class OpType>
// Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
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
//         return detail::dispatch_reduction<T, OpType>(input, normalized_axes, keepdim, stream);
//     });
// }

// // ✅ Mean dispatcher for BASIC operations
// template <template <typename> class SumOpType>
// Tensor _reduce_mean_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_mean_kernel<T, SumOpType>(input, normalized_axes, keepdim, stream);
//     });
// }

// ✅ Mean dispatcher for NaN-AWARE operations (only for floating point)
// template <template <typename> class SumOpType>
// Tensor _reduce_dispatcher(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
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
//         return detail::dispatch_mean_kernel<T, SumOpType>(input, normalized_axes, keepdim, stream);
//     });
// }

} // anonymous namespace

// =================================================================
// 1. Core Reductions (All types supported)
// =================================================================
Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, SumOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_product(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ProductOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}
Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, MinOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}
Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, MaxOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_mean_kernel<T, SumOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

// =================================================================
// 2. NaN-Aware Reductions (FLOATING POINT ONLY)
// =================================================================
Tensor reduce_nansum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanSumOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_nanproduct(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanProductOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_nanmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanMinOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_nanmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanMaxOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_nanmean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

// =================================================================
// 3. Index Reductions (All types supported)
// =================================================================
Tensor reduce_argmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ArgMinOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_argmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ArgMaxOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

// =================================================================
// 4. NaN-Aware Index Reductions (FLOATING POINT ONLY)
// =================================================================
Tensor reduce_nanargmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanArgMinOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}

Tensor reduce_nanargmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanArgMaxOp>(input, normalized_axes, keepdim, stream);//✨✨✨
    });
}
//==================================================
// VARIANCE OPERATIONS
//==================================================

Tensor reduce_var(const Tensor& input, 
                       const std::vector<int64_t>& axes, 
                       bool keepdim, 
                       int64_t correction, cudaStream_t stream) {//✨✨✨   
    // âœ… VALIDATION 1: Parameter bounds check (happens once per API call)
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_variance: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_variance_kernel<T, VarianceOp>(
            input, normalized_axes, keepdim, correction, stream//✨✨✨
        );
    });
}

Tensor reduce_nanvar(const Tensor& input, 
                          const std::vector<int64_t>& axes, 
                          bool keepdim, 
                          int64_t correction, cudaStream_t stream) {//✨✨✨
    // âœ… VALIDATION 1: Parameter bounds check
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_nanvariance: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_variance_kernel<T, NanVarianceOp>(
            input, normalized_axes, keepdim, correction, stream
        ); //✨✨✨
    });
}

Tensor reduce_std(const Tensor& input, 
                  const std::vector<int64_t>& axes, 
                  bool keepdim, 
                  int64_t correction, cudaStream_t stream) {//✨✨✨
    // âœ… VALIDATION 1: Parameter bounds check
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_std: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    // Compute variance first (validation already done above)
    Tensor var = reduce_var(input, axes, keepdim, correction, stream);
    
    // Apply element-wise sqrt (TODO: implement sqrt unary op)
    return sqrt(var,stream);//✨✨✨
}

Tensor reduce_nanstd(const Tensor& input, 
                     const std::vector<int64_t>& axes, 
                     bool keepdim, 
                     int64_t correction, cudaStream_t stream) {//✨✨✨ 
    // âœ… VALIDATION 1: Parameter bounds check
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_nanstd: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }

    Tensor var = reduce_nanvar(input, axes, keepdim, correction, stream );
    return sqrt(var,stream);//✨✨✨
}

//==================================================
// COMBINED STATISTICS
//==================================================
//==================================================
// COMBINED STATISTICS (Efficient single-pass computation)
//==================================================

std::pair<Tensor, Tensor> reduce_var_mean(const Tensor& input, 
                                          const std::vector<int64_t>& axes, 
                                          bool keepdim, 
                                          int64_t correction, cudaStream_t stream) {//✨✨✨
    // âœ… VALIDATION 1: Parameter bounds check
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_var_mean: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    // Compute mean first (always with keepdim=true for variance computation)
    Tensor mean_for_variance = reduce_mean(input, axes, true, stream);
    
    // Compute variance using the mean
    Tensor var = dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_variance_kernel<T, VarianceOp>(
            input, normalized_axes, keepdim, correction, stream//✨✨✨
        );
    });
    
    // Compute mean again with correct keepdim setting for output
    // (Small performance cost, but avoids needing squeeze() function)
    Tensor mean = reduce_mean(input, axes, keepdim, stream);
    
    return std::make_pair(var, mean);
}

std::pair<Tensor, Tensor> reduce_std_mean(const Tensor& input, 
                                          const std::vector<int64_t>& axes, 
                                          bool keepdim, 
                                          int64_t correction, cudaStream_t stream) {//✨✨✨    
    // âœ… VALIDATION 1: Parameter bounds check
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_std_mean: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    // Reuse var_mean, then take sqrt of variance
    auto [var, mean] = reduce_var_mean(input, axes, keepdim, correction, stream);
    Tensor std = sqrt(var, stream);  // Element-wise sqrt
    
    return std::make_pair(std, mean);
}
// std::pair<Tensor, Tensor> reduce_var_mean(const Tensor& input, 
//                                           const std::vector<int64_t>& axes, 
//                                           bool keepdim, 
//                                           int64_t correction) {
//     // âœ… VALIDATION 1: Parameter bounds check
//     if (correction < 0) {
//         throw std::runtime_error(
//             "reduce_var_mean: correction must be non-negative, got " + 
//             std::to_string(correction)
//         );
//     }
    
//     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
//     // Compute mean first (always with keepdim=true for variance computation)
//     Tensor mean = reduce_mean(input, axes, true);
    
//     // Compute variance using the mean
//     Tensor var = dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
//         using T = decltype(T_val);
//         return detail::dispatch_variance_kernel<T, VarianceOp>(
//             input, normalized_axes, keepdim, correction
//         );
//     });
    
//     // Adjust mean dimensions if keepdim=false
//     if (!keepdim) {
//         mean = mean.squeeze(axes);
//     }
    
//     return std::make_pair(var, mean);
// }

// std::pair<Tensor, Tensor> reduce_std_mean(const Tensor& input, 
//                                           const std::vector<int64_t>& axes, 
//                                           bool keepdim, 
//                                           int64_t correction) {
//     // âœ… VALIDATION 1: Parameter bounds check
//     if (correction < 0) {
//         throw std::runtime_error(
//             "reduce_std_mean: correction must be non-negative, got " + 
//             std::to_string(correction)
//         );
//     }
    
//     auto [var, mean] = reduce_var_mean(input, axes, keepdim, correction);
//     Tensor std = sqrt(var);
    
//     return std::make_pair(std, mean);
// }

} // namespace OwnTensor