// src/UnaryOps/Reduction.cpp - FIXED: Proper mean reuse in variance computation
#include "ops/UnaryOps/Reduction.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/ReductionUtils.h"
#include "ops/helpers/ReductionImpl.h"
#include "dtype/DtypeTraits.h"
#include <driver_types.h> //✨✨✨
#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <type_traits>
#include "ops/UnaryOps/Arithmetics.h" //for sqrt()

namespace OwnTensor {
using namespace detail;


// =================================================================
// 1. Core Reductions (All types supported)
// =================================================================
Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, SumOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_product(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ProductOp>(input, normalized_axes, keepdim, stream);       
    });
}

Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, MinOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, MaxOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_mean_kernel<T, SumOp>(input, normalized_axes, keepdim, stream);
    });
}

// =================================================================
// 2. NaN-Aware Reductions (FLOATING POINT ONLY)
// =================================================================
Tensor reduce_nansum(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nansum: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_sum() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanSumOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_nanproduct(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
     if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanproduct: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_product() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanProductOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_nanmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanmin: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_min() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanMinOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_nanmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanmax: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_max() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanMaxOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_nanmean(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanmean: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_mean() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, keepdim, stream);
    });
}

// =================================================================
// 3. Index Reductions (All types supported)
// =================================================================
Tensor reduce_argmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
     std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ArgMinOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_argmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, ArgMaxOp>(input, normalized_axes, keepdim, stream);
    });
}

// =================================================================
// 4. NaN-Aware Index Reductions (FLOATING POINT ONLY)
// =================================================================
Tensor reduce_nanargmin(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanargmin: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_argmin() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanArgMinOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_nanargmax(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim, cudaStream_t stream) {
     if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanargmax: Bool dtype not supported. "
            "NaN-aware operations require floating-point types. "
            "Use reduce_argmax() instead."
        );
    }
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);

    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, NanArgMaxOp>(input, normalized_axes, keepdim, stream);
    });
}

//==================================================
// VARIANCE OPERATIONS
//==================================================

Tensor reduce_var(const Tensor& input, 
                       const std::vector<int64_t>& axes, 
                       bool keepdim, 
                       int64_t correction, cudaStream_t stream) {//✨✨✨   
    // if (input.dtype() == Dtype::Bool) {
    //     throw std::runtime_error(
    //         "reduce_var: Bool dtype not supported. "
    //     );
    // }
    // VALIDATION 1: Parameter bounds check (happens once per API call)
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_variance: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        if constexpr (std::is_same_v<T, bool>) {
            throw std::runtime_error("Bool reduction not supported");
        } else {
            
                return detail::dispatch_variance_kernel<T, VarianceOp>(input, normalized_axes, keepdim,correction, stream); //✨✨✨
            
                
        }
        // return detail::dispatch_variance_kernel<T, VarianceOp>(
        //     input, normalized_axes, keepdim, correction, stream//✨✨✨
        // );
    });
}

Tensor reduce_nanvar(const Tensor& input, 
                          const std::vector<int64_t>& axes, 
                          bool keepdim, 
                          int64_t correction, cudaStream_t stream) {//✨✨✨
                            if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanvar: Bool dtype not supported. "
            
        );
    }
    // VALIDATION 1: Parameter bounds check (happens once per API call)
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
        );//✨✨✨
    });
}

Tensor reduce_std(const Tensor& input, 
                  const std::vector<int64_t>& axes, 
                  bool keepdim, 
                  int64_t correction, cudaStream_t stream) {//✨✨✨
                     if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_std: Bool dtype not supported. "
        );
    }
    // VALIDATION 1: Parameter bounds check (happens once per API call)
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_std: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
     // Compute variance first (validation already done above)
    Tensor var = reduce_var(input, axes, keepdim, correction, stream);
    // Apply element-wise sqrt (TODO: implement sqrt unary op)
    return sqrt(var, stream);//✨✨✨
}

Tensor reduce_nanstd(const Tensor& input, 
                     const std::vector<int64_t>& axes, 
                     bool keepdim, 
                     int64_t correction, cudaStream_t stream) {//✨✨✨ 
                                            if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_nanstd: Bool dtype not supported. "
            
        );
    }
    // VALIDATION 1: Parameter bounds check (happens once per API call)
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_nanstd: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }

    Tensor var = reduce_nanvar(input, axes, keepdim, correction, stream);
    return sqrt(var, stream);//✨✨✨
}


//==================================================
// COMBINED STATISTICS - FIXED VERSION
//==================================================

std::pair<Tensor, Tensor> reduce_var_mean(const Tensor& input, 
                                          const std::vector<int64_t>& axes, 
                                          bool keepdim, 
                                          int64_t correction, cudaStream_t stream) {
                                                                if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_var_mean: Bool dtype not supported. "
            "reduce_nanvar: Bool dtype not supported. "
            "Use reduce_mean() instead."
        );
    }
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_var_mean: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    // ✅ FIX: Compute mean ONCE with the correct keepdim setting
    Tensor mean = reduce_mean(input, axes, keepdim, stream);
    
    // ✅ FIX: Compute variance using dispatch_variance_kernel which will compute its own mean internally
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    Tensor var = dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_variance_kernel<T, VarianceOp>(
            input, normalized_axes, keepdim, correction, stream
        );
    });
    
    return std::make_pair(var, mean);
}

std::pair<Tensor, Tensor> reduce_std_mean(const Tensor& input, 
                                          const std::vector<int64_t>& axes, 
                                          bool keepdim, 
                                          int64_t correction, cudaStream_t stream) {
                                            if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "reduce_std_mean: Bool dtype not supported. "
            "reduce_nanstd: Bool dtype not supported. "
            "Use reduce_mean() instead."
        );
    }
    if (correction < 0) {
        throw std::runtime_error(
            "reduce_std_mean: correction must be non-negative, got " + 
            std::to_string(correction)
        );
    }
    
    auto [var, mean] = reduce_var_mean(input, axes, keepdim, correction, stream);
    Tensor std = sqrt(var, stream);
    
    return std::make_pair(std, mean);
}


// =================================================================
// 5. Boolean Reductions (Bool dtype only)
// =================================================================

Tensor reduce_all(const Tensor& input, const std::vector<int64_t>& axes, 
                  bool keepdim, cudaStream_t stream) {
    // Validate dtype
    // if (input.dtype() != Dtype::Bool) {
    //     throw std::runtime_error(
    //         "reduce_all: requires Bool dtype, got " + get_dtype_name(input.dtype()) +
    //         ". Use comparison operators (==, <, >, etc.) to create boolean tensors."
    //     );
    // }
    
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, AllOp>(input, normalized_axes, keepdim, stream);
    });
}

Tensor reduce_any(const Tensor& input, const std::vector<int64_t>& axes, 
                  bool keepdim, cudaStream_t stream) {
    // Validate dtype
    // if (input.dtype() != Dtype::Bool) {
    //     throw std::runtime_error(
    //         "reduce_any: requires Bool dtype, got " + get_dtype_name(input.dtype()) +
    //         ". Use comparison operators (==, <, >, etc.) to create boolean tensors."
    //     );
    // }
    
    std::vector<int64_t> normalized_axes = detail::normalize_axes(input.shape().dims, axes);
    
    return dispatch_by_dtype(input.dtype(), [&](auto T_val) -> Tensor {
        using T = decltype(T_val);
        return detail::dispatch_reduction<T, AnyOp>(input, normalized_axes, keepdim, stream);
    });
}
} // namespace OwnTensor


