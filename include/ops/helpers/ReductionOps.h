// include/ops/helpers/ReductionOps.h - FIXED: Uses GPU intrinsics
#pragma once

#ifndef OWNTENSOR_REDUCTION_OPS_H
#define OWNTENSOR_REDUCTION_OPS_H

// ═══════════════════════════════════════════════════════════
// COMPILATION CONTEXT SETUP
// ═══════════════════════════════════════════════════════════

#ifdef __CUDACC__
    // GPU COMPILATION (nvcc)
    #define DEVICE_HOST __device__ __host__
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <math.h>
    
    #ifndef CUDART_INF_F
        #define CUDART_INF_F __int_as_float(0x7f800000)
    #endif
    #ifndef CUDART_INF
        #define CUDART_INF __longlong_as_double(0x7ff0000000000000LL)
    #endif
#else
    // CPU COMPILATION (g++)
    #define DEVICE_HOST
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __host__
        #define __host__
    #endif
    
    #ifndef CUDART_INF_F
        #define CUDART_INF_F __builtin_huge_valf()
    #endif
    #ifndef CUDART_INF
        #define CUDART_INF __builtin_huge_val()
    #endif
#endif

#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <cstdint>

namespace OwnTensor {
namespace detail {

// ═══════════════════════════════════════════════════════════
// GPU INTRINSIC HELPERS (FORWARD DECLARATIONS)
// ═══════════════════════════════════════════════════════════

#ifdef __CUDA_ARCH__
// ✅ GPU device code - use intrinsics
template<typename T> __device__ inline T gpu_add(T a, T b) { return a + b; }
template<> __device__ inline __half gpu_add(__half a, __half b) { return __hadd(a, b); }
template<> __device__ inline __nv_bfloat16 gpu_add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }

template<typename T> __device__ inline T gpu_mul(T a, T b) { return a * b; }
template<> __device__ inline __half gpu_mul(__half a, __half b) { return __hmul(a, b); }
template<> __device__ inline __nv_bfloat16 gpu_mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }

template<typename T> __device__ inline bool gpu_lt(T a, T b) { return a < b; }
template<> __device__ inline bool gpu_lt(__half a, __half b) { return __hlt(a, b); }
template<> __device__ inline bool gpu_lt(__nv_bfloat16 a, __nv_bfloat16 b) { return __hlt(a, b); }

template<typename T> __device__ inline bool gpu_gt(T a, T b) { return a > b; }
template<> __device__ inline bool gpu_gt(__half a, __half b) { return __hgt(a, b); }
template<> __device__ inline bool gpu_gt(__nv_bfloat16 a, __nv_bfloat16 b) { return __hgt(a, b); }

template<typename T> __device__ inline bool gpu_isnan(T val) { return isnan(val); }
template<> __device__ inline bool gpu_isnan(__half val) { return __hisnan(val); }
template<> __device__ inline bool gpu_isnan(__nv_bfloat16 val) { return __hisnan(val); }

// #else
// // ✅ CPU host code - use regular operations
// template<typename T> inline T gpu_add(T a, T b) { return a + b; }
// template<typename T> inline T gpu_mul(T a, T b) { return a * b; }
// template<typename T> inline bool gpu_lt(T a, T b) { return a < b; }
// template<typename T> inline bool gpu_gt(T a, T b) { return a > b; }
// template<typename T> inline bool gpu_isnan(T val) { 
//     if constexpr (std::is_floating_point_v<T>) {
//         return std::isnan(val);
//     }
//     return false;
// }
#endif

// ═══════════════════════════════════════════════════════════
// HELPER TRAITS
// ═══════════════════════════════════════════════════════════

template <typename T>
constexpr bool is_half_float_v = std::is_same_v<T, bfloat16_t> || 
                                 std::is_same_v<T, float16_t>;

#ifdef __CUDACC__
template <typename T>
constexpr bool is_native_half_v = std::is_same_v<T, __half> || 
                                  std::is_same_v<T, __nv_bfloat16>;
#else
template <typename T>
constexpr bool is_native_half_v = false;
#endif

template <typename T>
constexpr bool is_any_float_v = std::is_floating_point_v<T> || 
                                is_half_float_v<T> || 
                                is_native_half_v<T>;

// ═══════════════════════════════════════════════════════════
// VALUE-INDEX PAIR FOR ARG REDUCTIONS
// ═══════════════════════════════════════════════════════════

template <typename T>
struct ValueIndex {
    T value;
    int64_t index;

    DEVICE_HOST ValueIndex() : value(T{}), index(-1) {}
    DEVICE_HOST ValueIndex(T val, int64_t idx) : value(val), index(idx) {}

    DEVICE_HOST bool operator>(const ValueIndex<T>& other) const {
        #ifdef __CUDA_ARCH__
        return gpu_gt(value, other.value);
        #else
        return value > other.value;
        #endif
    }
    
    DEVICE_HOST bool operator<(const ValueIndex<T>& other) const {
        #ifdef __CUDA_ARCH__
        return gpu_lt(value, other.value);
        #else
        return value < other.value;
        #endif
    }
};

// ═══════════════════════════════════════════════════════════
// DEVICE-SAFE HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════

template <typename T>
DEVICE_HOST constexpr T get_lowest_value() {
    if constexpr (std::is_same_v<T, float16_t>) {
        return T(-65504.0f);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return T(-3.38953e38f);
    }
#ifdef __CUDACC__
    else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(-65504.0f);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(-3.38953e38f);
    }
#endif
    else if constexpr (std::is_same_v<T, float>) {
        return -3.402823466e+38f;
    } else if constexpr (std::is_same_v<T, double>) {
        return -1.7976931348623158e+308;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return -32768;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return -2147483648;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return -9223372036854775807LL - 1LL;
    }
    return T{};
}

template <typename T>
DEVICE_HOST constexpr T get_max_value() {
    if constexpr (std::is_same_v<T, float16_t>) {
        return T(65504.0f);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return T(3.38953e38f);
    }
#ifdef __CUDACC__
    else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(65504.0f);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(3.38953e38f);
    }
#endif
    else if constexpr (std::is_same_v<T, float>) {
        return 3.402823466e+38f;
    } else if constexpr (std::is_same_v<T, double>) {
        return 1.7976931348623158e+308;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return 32767;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return 2147483647;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return 9223372036854775807LL;
    }
    return T{};
}

template <typename T>
DEVICE_HOST inline bool is_nan_check(T val) {
    #ifdef __CUDA_ARCH__
    return gpu_isnan(val);
    #else
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(val);
    } else if constexpr (is_half_float_v<T>) {
        float f_val = static_cast<float>(val);
        return std::isnan(f_val);
    }
    return false;
    #endif
}

// ═══════════════════════════════════════════════════════════
// ACCUMULATOR TYPE SELECTOR
// ═══════════════════════════════════════════════════════════

template<typename T>
struct AccumulatorTypeSelector {
    using type = T;
};
// Integer types use int64_t to prevent overflow
template<> struct AccumulatorTypeSelector<int16_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int32_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int64_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint16_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint32_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint64_t> { using type = int64_t; };
// Half precision floats use float for better precision
template<> struct AccumulatorTypeSelector<float16_t> { using type = float; };
template<> struct AccumulatorTypeSelector<bfloat16_t> { using type = float; };
// ✅ FIX: Bool should accumulate as int64_t (like PyTorch/NumPy)
template<> struct AccumulatorTypeSelector<bool> { using type = int64_t; };


#ifdef __CUDACC__
template<> struct AccumulatorTypeSelector<__half> { using type = float; };
template<> struct AccumulatorTypeSelector<__nv_bfloat16> { using type = float; };
#endif

template<typename T>
using AccumulatorType = typename AccumulatorTypeSelector<T>::type;

// ═══════════════════════════════════════════════════════════
// ✅ CORE REDUCTION OPERATIONS (NOW USES GPU INTRINSICS!)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct SumOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        return AccT(0); 
    }
   
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        
        #ifdef __CUDA_ARCH__
        // ✅ GPU: Use intrinsics for half types
        // constexpr bool is_bool = std::is_same_v<AccT, bool>;
        // if constexpr (is_bool) {
        //     throw std::runtime_error(
        //     "Sum reduction is not supported for Bool type."
        // );    
        // }
        if constexpr (is_any_float_v<AccT>) {
            if (gpu_isnan(a)) return a;
            if (gpu_isnan(b)) return b;
        }
        return gpu_add(a, b);
        #else
        // CPU: Regular addition
          if constexpr (is_any_float_v<AccT>) {
            if (is_nan_check(a)) return a;
            if (is_nan_check(b)) return b;
        }
        return a + b;
        #endif
    }
};

template <typename T>
struct ProductOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        return AccT(1); 
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        #ifdef __CUDA_ARCH__
        // ✅ GPU path
        if constexpr (is_any_float_v<AccT>) {
            if (gpu_isnan(a)) return a;
            if (gpu_isnan(b)) return b;
        }
        return gpu_mul(a, b);
        #else
        // CPU path
        if constexpr (is_any_float_v<AccT>) {
            if (is_nan_check(a)) return a;
            if (is_nan_check(b)) return b;
        }
        return a * b;
        #endif
    }
};

template <typename T>
struct MinOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        }
#ifdef __CUDACC__
        else if constexpr (is_native_half_v<T>) {
            return get_max_value<T>();
        }
#endif
        else {
            return get_max_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        #ifdef __CUDA_ARCH__
        // ✅ GPU: Use intrinsics
        if constexpr (is_any_float_v<T>) {
            if (gpu_isnan(a)) return a;
            if (gpu_isnan(b)) return b;
        }
        return gpu_lt(a, b) ? a : b;
        #else
        // CPU path
        if constexpr (is_any_float_v<T>) {
            if (is_nan_check(a)) return a;
            if (is_nan_check(b)) return b;
        }
        return (a < b) ? a : b;
        #endif
    }
};

template <typename T>
struct MaxOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        }
#ifdef __CUDACC__
        else if constexpr (is_native_half_v<T>) {
            return get_lowest_value<T>();
        }
#endif
        else {
            return get_lowest_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
        // ✅ GPU: Use intrinsics
        if constexpr (is_any_float_v<T>) {
            if (gpu_isnan(a)) return a;
            if (gpu_isnan(b)) return b;
        }
        return gpu_gt(a, b) ? a : b;
        #else
        // CPU path
        if constexpr (is_any_float_v<T>) {
            if (is_nan_check(a)) return a;
            if (is_nan_check(b)) return b;
        }
        return (a > b) ? a : b;
        #endif
    }
};
// ═══════════════════════════════════════════════════════════
// VARIANCE OPERATION (Two-pass algorithm for numerical stability)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct VarianceOp {
    using AccT = AccumulatorType<T>;
    int64_t correction;  // Bessel's correction
    AccT mean_value;     // Pre-computed mean
    
    DEVICE_HOST explicit VarianceOp(int64_t corr = 1, AccT mean = AccT(0)) 
        : correction(corr), mean_value(mean) {}
    
    DEVICE_HOST AccT identity() const { return AccT(0); }
    
    DEVICE_HOST AccT reduce(const AccT& acc, const AccT& val) const {
        #ifdef __CUDA_ARCH__
        // ✅ GPU: Propagate NaN immediately
        if constexpr (is_any_float_v<AccT>) {
            if (gpu_isnan(acc)) return acc;  // Already NaN, propagate it
            if (gpu_isnan(val)) return val;  // New NaN, propagate it
        }
        AccT diff = val - mean_value;
        return gpu_add(acc, gpu_mul(diff, diff));
        #else
        // ✅ CPU: Propagate NaN immediately
        if constexpr (is_any_float_v<AccT>) {
            if (is_nan_check(acc)) return acc;  // Already NaN, propagate it
            if (is_nan_check(val)) return val;  // New NaN, propagate it
        }
        AccT diff = val - mean_value;
        return acc + diff * diff;
        #endif
    }
};
// ═══════════════════════════════════════════════════════════
// NaN-AWARE OPERATIONS (ALSO USE GPU INTRINSICS)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct NanSumOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { return AccT(0); }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
        if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
            if (gpu_isnan(a)) return b;
            if (gpu_isnan(b)) return a;
        }
        return gpu_add(a, b);
        #else
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return a + b;
        #endif
    }
};

template <typename T>
struct NanProductOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { return AccT(1); }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
        if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
            if (gpu_isnan(a)) return b;
            if (gpu_isnan(b)) return a;
        }
        return gpu_mul(a, b);
        #else
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return a * b;
        #endif
    }
};

template <typename T>
struct NanMinOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        }
#ifdef __CUDACC__
        else if constexpr (is_native_half_v<T>) {
            return get_max_value<T>();
        }
#endif
        else {
            return get_max_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
        if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
            if (gpu_isnan(a)) return b;
            if (gpu_isnan(b)) return a;
        }
        return gpu_lt(a, b) ? a : b;
        #else
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return (a < b) ? a : b;
        #endif
    }
};

template <typename T>
struct NanMaxOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        }
#ifdef __CUDACC__
        else if constexpr (is_native_half_v<T>) {
            return get_lowest_value<T>();
        }
#endif
        else {
            return get_lowest_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
        if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
            if (gpu_isnan(a)) return b;
            if (gpu_isnan(b)) return a;
        }
        return gpu_gt(a, b) ? a : b;
        #else
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return (a > b) ? a : b;
        #endif
    }
};// ═══════════════════════════════════════════════════════════
// NaN-aware variance (IGNORES NaNs, doesn't propagate them)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct NanVarianceOp {
    using AccT = AccumulatorType<T>;
    int64_t correction;
    AccT mean_value;
    
    DEVICE_HOST explicit NanVarianceOp(int64_t corr = 1, AccT mean = AccT(0)) 
        : correction(corr), mean_value(mean) {}
    
    DEVICE_HOST AccT identity() const { return AccT(0); }
    
    DEVICE_HOST AccT reduce(const AccT& acc, const AccT& val) const {
        #ifdef __CUDA_ARCH__
        // ✅ GPU: Skip NaN values (don't propagate them)
        if (gpu_isnan(val)) return acc;  // Ignore NaN, return accumulator unchanged
        AccT diff = val - mean_value;
        return gpu_add(acc, gpu_mul(diff, diff));
        #else
        // ✅ CPU: Skip NaN values (don't propagate them)
        if (is_nan_check(val)) return acc;  // Ignore NaN, return accumulator unchanged
        AccT diff = val - mean_value;
        return acc + diff * diff;
        #endif
    }
};
// ═══════════════════════════════════════════════════════════
// INDEX REDUCTIONS (ArgMin/ArgMax) - ALSO USE GPU INTRINSICS
// ═══════════════════════════════════════════════════════════

template <typename T>
struct ArgMinOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const { 
        return ValueIndex<T>(get_max_value<T>(), -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        #ifdef __CUDA_ARCH__
        if constexpr (is_any_float_v<T>) {
            if (gpu_isnan(a.value)) return a;
            if (gpu_isnan(b.value)) return b;
        }
        if (gpu_lt(a.value, b.value)) {
            return a;
        } else if (gpu_gt(a.value, b.value)) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #else
        if constexpr (is_any_float_v<T>) {
            if (is_nan_check(a.value)) return a;
            if (is_nan_check(b.value)) return b;
        }
        if (a.value < b.value) {
            return a;
        } else if (b.value < a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #endif
    }
};

template <typename T>
struct ArgMaxOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const {
        return ValueIndex<T>(get_lowest_value<T>(), -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        #ifdef __CUDA_ARCH__
        if constexpr (is_any_float_v<T>) {
            if (gpu_isnan(a.value)) return a;
            if (gpu_isnan(b.value)) return b;
        }
        if (gpu_gt(a.value, b.value)) {
            return a;
        } else if (gpu_lt(a.value, b.value)) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #else
        if constexpr (is_any_float_v<T>) {
            if (is_nan_check(a.value)) return a;
            if (is_nan_check(b.value)) return b;
        }
        if (a.value > b.value) {
            return a;
        } else if (b.value > a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #endif
    }
};

template <typename T>
struct NanArgMinOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const { 
        return ValueIndex<T>(get_max_value<T>(), -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        #ifdef __CUDA_ARCH__
        const bool a_is_nan = gpu_isnan(a.value);
        const bool b_is_nan = gpu_isnan(b.value);
        if (a_is_nan && b_is_nan) return (a.index < b.index) ? a : b;
        if (a_is_nan) return b;
        if (b_is_nan) return a;
        
        if (gpu_lt(a.value, b.value)) {
            return a;
        } else if (gpu_gt(a.value, b.value)) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #else
        const bool a_is_nan = is_nan_check(a.value);
        const bool b_is_nan = is_nan_check(b.value);
        if (a_is_nan && b_is_nan) return (a.index < b.index) ? a : b;
        if (a_is_nan) return b;
        if (b_is_nan) return a;
        
        if (a.value < b.value) {
            return a;
        } else if (b.value < a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #endif
    }
};

template <typename T>
struct NanArgMaxOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const {
        return ValueIndex<T>(get_lowest_value<T>(), -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        #ifdef __CUDA_ARCH__
        const bool a_is_nan = gpu_isnan(a.value);
        const bool b_is_nan = gpu_isnan(b.value);
        if (a_is_nan && b_is_nan) return (a.index < b.index) ? a : b;
        if (a_is_nan) return b;
        if (b_is_nan) return a;
        
        if (gpu_gt(a.value, b.value)) {
            return a;
        } else if (gpu_lt(a.value, b.value)) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #else
        const bool a_is_nan = is_nan_check(a.value);
        const bool b_is_nan = is_nan_check(b.value);
        if (a_is_nan && b_is_nan) return (a.index < b.index) ? a : b;
        if (a_is_nan) return b;
        if (b_is_nan) return a;
        
        if (a.value > b.value) {
            return a;
        } else if (b.value > a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
        #endif
    }
};



// ═══════════════════════════════════════════════════════════
// BOOLEAN REDUCTION OPERATIONS (Bool dtype only)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct AllOp {
    using AccT = bool;  // Always accumulate as bool
    
    DEVICE_HOST bool identity() const { 
        return true;  // Neutral element for AND operation
    }
    
    DEVICE_HOST bool reduce(const bool& a, const bool& b) const {
        return a && b;  // Logical AND
    }
};

template <typename T>
struct AnyOp {
    using AccT = bool;  // Always accumulate as bool
    
    DEVICE_HOST bool identity() const { 
        return false;  // Neutral element for OR operation
    }
    
    DEVICE_HOST bool reduce(const bool& a, const bool& b) const {
        return a || b;  // Logical OR
    }
};
// ═══════════════════════════════════════════════════════════
// REDUCTION TYPE DISPATCHER
// ═══════════════════════════════════════════════════════════

enum class ReductionType {
    SUM,
    PRODUCT,
    MIN,
    MAX,
    NANSUM,
    NANPRODUCT,
    NANMIN,
    NANMAX,
    ARGMIN,
    ARGMAX,
    NANARGMIN,
    NANARGMAX,
     ALL,     
    ANY   
};

template<ReductionType R, typename T>
struct ReductionOpSelector;

template<typename T> struct ReductionOpSelector<ReductionType::SUM, T> { using type = SumOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::PRODUCT, T> { using type = ProductOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::MIN, T> { using type = MinOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::MAX, T> { using type = MaxOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::NANSUM, T> { using type = NanSumOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::NANPRODUCT, T> { using type = NanProductOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::NANMIN, T> { using type = NanMinOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::NANMAX, T> { using type = NanMaxOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::ARGMIN, T> { using type = ArgMinOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::ARGMAX, T> { using type = ArgMaxOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::NANARGMIN, T> { using type = NanArgMinOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::NANARGMAX, T> { using type = NanArgMaxOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::ALL, T> { using type = AllOp<T>; };
template<typename T> struct ReductionOpSelector<ReductionType::ANY, T> { using type = AnyOp<T>; };
} // namespace detail
} // namespace OwnTensor

#endif // OWNTENSOR_REDUCTION_OPS_H