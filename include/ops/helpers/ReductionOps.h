// include/utilities/ReductionOps.h - COMPLETE GPU INTRINSIC OPTIMIZED VERSION
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
    
    namespace OwnTensor {
        using float16_t = __half;
        using bfloat16_t = __nv_bfloat16;
    }
    
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
    
    #include "dtype/Types.h"
    
    #ifndef CUDART_INF_F
        #define CUDART_INF_F __builtin_huge_valf()
    #endif
    #ifndef CUDART_INF
        #define CUDART_INF __builtin_huge_val()
    #endif
#endif

#include <limits>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <cstdint>

namespace OwnTensor {
namespace detail {

// ═══════════════════════════════════════════════════════════
// HELPER TRAITS
// ═══════════════════════════════════════════════════════════

template <typename T>
constexpr bool is_half_float_v = std::is_same_v<T, bfloat16_t> || 
                                 std::is_same_v<T, float16_t>;

template <typename T>
constexpr bool is_any_float_v = std::is_floating_point_v<T> || is_half_float_v<T>;

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
        return value > other.value;
    }
    DEVICE_HOST bool operator<(const ValueIndex<T>& other) const {
        return value < other.value;
    }
};

// ═══════════════════════════════════════════════════════════
// HELPER FUNCTIONS (GPU-COMPATIBLE)
// ═══════════════════════════════════════════════════════════

template <typename T>
DEVICE_HOST T get_lowest_value() {
    if constexpr (std::is_same_v<T, float16_t>) {
        #ifdef __CUDA_ARCH__
            return __float2half(-65504.0f);
        #else
            return T(-65504.0f);
        #endif
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        #ifdef __CUDA_ARCH__
            return __float2bfloat16(-3.38953e38f);
        #else
            return T(-3.38953e38f);
        #endif
    } else if constexpr (std::is_arithmetic_v<T>) {
        return std::numeric_limits<T>::lowest();
    }
    return T{};
}

template <typename T>
DEVICE_HOST T get_max_value() {
    if constexpr (std::is_same_v<T, float16_t>) {
        #ifdef __CUDA_ARCH__
            return __float2half(65504.0f);
        #else
            return T(65504.0f);
        #endif
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        #ifdef __CUDA_ARCH__
            return __float2bfloat16(3.38953e38f);
        #else
            return T(3.38953e38f);
        #endif
    } else if constexpr (std::is_arithmetic_v<T>) {
        return std::numeric_limits<T>::max();
    }
    return T{};
}

template <typename T>
DEVICE_HOST inline bool is_nan_check(T val) {
    if constexpr (std::is_floating_point_v<T>) {
        #ifdef __CUDA_ARCH__
            return isnan(val);
        #else
            return std::isnan(val);
        #endif
    } else if constexpr (is_half_float_v<T>) {
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_same_v<T, float16_t>) {
                return __hisnan(val);
            } else {
                return __hisnan(__bfloat162half(val));
            }
        #else
            float f_val = static_cast<float>(val);
            return std::isnan(f_val);
        #endif
    }
    return false;  
}

// ═══════════════════════════════════════════════════════════
// ACCUMULATOR TYPE SELECTOR
// ═══════════════════════════════════════════════════════════

template<typename T>
struct AccumulatorTypeSelector {
    using type = T;
};

// Integers accumulate in int64_t
template<> struct AccumulatorTypeSelector<int16_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int32_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int64_t> { using type = int64_t; };

// CRITICAL FIX: Add unsigned integer support
template<> struct AccumulatorTypeSelector<uint16_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint32_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint64_t> { using type = int64_t; };

// FP16/BF16 accumulate in float for better precision
template<> struct AccumulatorTypeSelector<float16_t> { using type = float; };
template<> struct AccumulatorTypeSelector<bfloat16_t> { using type = float; };

template<typename T>
using AccumulatorType = typename AccumulatorTypeSelector<T>::type;

// ═══════════════════════════════════════════════════════════
// ✅ NEW: GPU INTRINSIC WRAPPERS (TYPE-GENERIC)
// ═══════════════════════════════════════════════════════════

#ifdef __CUDA_ARCH__

// ✅ Addition intrinsics
__device__ inline float intrinsic_add(float a, float b) {
    return __fadd_rn(a, b);
}

__device__ inline double intrinsic_add(double a, double b) {
    return __dadd_rn(a, b);
}

__device__ inline float16_t intrinsic_add(float16_t a, float16_t b) {
    return __hadd(a, b);
}

#if __CUDA_ARCH__ >= 800  // Ampere+ for BF16
__device__ inline bfloat16_t intrinsic_add(bfloat16_t a, bfloat16_t b) {
    return __hadd(a, b);
}
#else
__device__ inline bfloat16_t intrinsic_add(bfloat16_t a, bfloat16_t b) {
    // Fallback: convert to float, add, convert back
    return __float2bfloat16(__fadd_rn(__bfloat162float(a), __bfloat162float(b)));
}
#endif

// ✅ Multiplication intrinsics
__device__ inline float intrinsic_mul(float a, float b) {
    return __fmul_rn(a, b);
}

__device__ inline double intrinsic_mul(double a, double b) {
    return __dmul_rn(a, b);
}

__device__ inline float16_t intrinsic_mul(float16_t a, float16_t b) {
    return __hmul(a, b);
}

#if __CUDA_ARCH__ >= 800
__device__ inline bfloat16_t intrinsic_mul(bfloat16_t a, bfloat16_t b) {
    return __hmul(a, b);
}
#else
__device__ inline bfloat16_t intrinsic_mul(bfloat16_t a, bfloat16_t b) {
    return __float2bfloat16(__fmul_rn(__bfloat162float(a), __bfloat162float(b)));
}
#endif

// ✅ Max intrinsics
__device__ inline float intrinsic_max(float a, float b) {
    return fmaxf(a, b);
}

__device__ inline double intrinsic_max(double a, double b) {
    return fmax(a, b);
}

__device__ inline float16_t intrinsic_max(float16_t a, float16_t b) {
    return __hmax(a, b);
}

#if __CUDA_ARCH__ >= 800
__device__ inline bfloat16_t intrinsic_max(bfloat16_t a, bfloat16_t b) {
    return __hmax(a, b);
}
#else
__device__ inline bfloat16_t intrinsic_max(bfloat16_t a, bfloat16_t b) {
    return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
}
#endif

// ✅ Min intrinsics
__device__ inline float intrinsic_min(float a, float b) {
    return fminf(a, b);
}

__device__ inline double intrinsic_min(double a, double b) {
    return fmin(a, b);
}

__device__ inline float16_t intrinsic_min(float16_t a, float16_t b) {
    return __hmin(a, b);
}

#if __CUDA_ARCH__ >= 800
__device__ inline bfloat16_t intrinsic_min(bfloat16_t a, bfloat16_t b) {
    return __hmin(a, b);
}
#else
__device__ inline bfloat16_t intrinsic_min(bfloat16_t a, bfloat16_t b) {
    return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
}
#endif

// ✅ FMA intrinsics (a*b + c)
__device__ inline float intrinsic_fma(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

__device__ inline double intrinsic_fma(double a, double b, double c) {
    return __fma_rn(a, b, c);
}

__device__ inline float16_t intrinsic_fma(float16_t a, float16_t b, float16_t c) {
    return __hfma(a, b, c);
}

#if __CUDA_ARCH__ >= 800
__device__ inline bfloat16_t intrinsic_fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
    return __hfma(a, b, c);
}
#else
__device__ inline bfloat16_t intrinsic_fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    float fc = __bfloat162float(c);
    return __float2bfloat16(__fmaf_rn(fa, fb, fc));
}
#endif

#endif // __CUDA_ARCH__

// ═══════════════════════════════════════════════════════════
// CORE REDUCTION OPERATIONS (NOW USING INTRINSICS)
// ═══════════════════════════════════════════════════════════

// --- Sum ---
template <typename T>
struct SumOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        return AccT(0); 
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        #ifdef __CUDA_ARCH__
            return intrinsic_add(a, b);  // ✅ Always use intrinsic on GPU
        #else
            return a + b;  // CPU path
        #endif
    }
};

// --- Product ---
template <typename T>
struct ProductOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        return AccT(1); 
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        #ifdef __CUDA_ARCH__
            return intrinsic_mul(a, b);  // ✅ Always use intrinsic on GPU
        #else
            return a * b;  // CPU path
        #endif
    }
};

// --- Min ---
template <typename T>
struct MinOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(std::numeric_limits<T>::max());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        } else {
            return std::numeric_limits<AccT>::max();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
                return intrinsic_min(a, b);  // ✅ Use intrinsic for floats
            } else {
                return (a < b) ? a : b;  // Integer min
            }
        #else
            return std::min(a, b);  // CPU path
        #endif
    }
};

// --- Max ---
template <typename T>
struct MaxOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(std::numeric_limits<T>::lowest());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        } else {
            return std::numeric_limits<AccT>::lowest();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
                return intrinsic_max(a, b);  // ✅ Use intrinsic for floats
            } else {
                return (a > b) ? a : b;  // Integer max
            }
        #else
            return std::max(a, b);  // CPU path
        #endif
    }
};

// ═══════════════════════════════════════════════════════════
// NaN-AWARE OPERATIONS (WITH INTRINSICS)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct NanSumOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { return AccT(0); }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_floating_point_v<AccT>) {
                if (isnan(a)) return b;
                if (isnan(b)) return a;
                return intrinsic_add(a, b);  // ✅ Use intrinsic after NaN check
            } else if constexpr (is_half_float_v<AccT>) {
                if (is_nan_check(a)) return b;
                if (is_nan_check(b)) return a;
                return intrinsic_add(a, b);  // ✅ Use intrinsic
            } else {
                return a + b;  // Integers can't be NaN
            }
        #else
            if constexpr (std::is_floating_point_v<AccT>) {
                if (std::isnan(a)) return b;
                if (std::isnan(b)) return a;
            }
            return a + b;  // CPU path
        #endif
    }
};

template <typename T>
struct NanProductOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { return AccT(1); }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_floating_point_v<AccT>) {
                if (isnan(a)) return b;
                if (isnan(b)) return a;
                return intrinsic_mul(a, b);  // ✅ Use intrinsic
            } else if constexpr (is_half_float_v<AccT>) {
                if (is_nan_check(a)) return b;
                if (is_nan_check(b)) return a;
                return intrinsic_mul(a, b);  // ✅ Use intrinsic
            } else {
                return a * b;
            }
        #else
            if constexpr (std::is_floating_point_v<AccT>) {
                if (std::isnan(a)) return b;
                if (std::isnan(b)) return a;
            }
            return a * b;  // CPU path
        #endif
    }
};

template <typename T>
struct NanMinOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(std::numeric_limits<T>::max());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        } else {
            return std::numeric_limits<AccT>::max();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_floating_point_v<AccT>) {
                if (isnan(a)) return b;
                if (isnan(b)) return a;
                return intrinsic_min(a, b);  // ✅ Use intrinsic
            } else if constexpr (is_half_float_v<AccT>) {
                if (is_nan_check(a)) return b;
                if (is_nan_check(b)) return a;
                return intrinsic_min(a, b);  // ✅ Use intrinsic
            } else {
                return (a < b) ? a : b;
            }
        #else
            if constexpr (std::is_floating_point_v<AccT>) {
                if (std::isnan(a)) return b;
                if (std::isnan(b)) return a;
                return std::min(a, b);
            }
            return std::min(a, b);  // CPU path
        #endif
    }
};

template <typename T>
struct NanMaxOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        if constexpr (std::is_integral_v<T>) {
            return static_cast<AccT>(std::numeric_limits<T>::lowest());
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        } else {
            return std::numeric_limits<AccT>::lowest();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        #ifdef __CUDA_ARCH__
            if constexpr (std::is_floating_point_v<AccT>) {
                if (isnan(a)) return b;
                if (isnan(b)) return a;
                return intrinsic_max(a, b);  // ✅ Use intrinsic
            } else if constexpr (is_half_float_v<AccT>) {
                if (is_nan_check(a)) return b;
                if (is_nan_check(b)) return a;
                return intrinsic_max(a, b);  // ✅ Use intrinsic
            } else {
                return (a > b) ? a : b;
            }
        #else
            if constexpr (std::is_floating_point_v<AccT>) {
                if (std::isnan(a)) return b;
                if (std::isnan(b)) return a;
                return std::max(a, b);
            }
            return std::max(a, b);  // CPU path
        #endif
    }
};

// ═══════════════════════════════════════════════════════════
// INDEX REDUCTIONS (ArgMin/ArgMax) - UNCHANGED
// ═══════════════════════════════════════════════════════════

template <typename T>
struct ArgMinOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const { 
        return ValueIndex<T>(get_max_value<T>(), -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        if constexpr (is_half_float_v<T>) {
            float a_val = static_cast<float>(a.value);
            float b_val = static_cast<float>(b.value);
            
            if (a_val < b_val) {
                return a;
            } else if (b_val < a_val) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        } else {
            if (a.value < b.value) {
                return a;
            } else if (b.value < a.value) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        }
    }
};

template <typename T>
struct ArgMaxOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const {
        T initial_val;
        if constexpr (is_any_float_v<T>) {
            #ifdef __CUDA_ARCH__
                if constexpr (std::is_same_v<T, float>) {
                    initial_val = -CUDART_INF_F;
                } else if constexpr (std::is_same_v<T, double>) {
                    initial_val = -CUDART_INF;
                } else {
                    initial_val = get_lowest_value<T>();
                }
            #else
                if constexpr (std::is_floating_point_v<T>) {
                    initial_val = -std::numeric_limits<T>::infinity();
                } else {
                    initial_val = get_lowest_value<T>();
                }
            #endif
        } else {
            initial_val = get_lowest_value<T>();
        }
        return ValueIndex<T>(initial_val, -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        if constexpr (is_half_float_v<T>) {
            float a_val = static_cast<float>(a.value);
            float b_val = static_cast<float>(b.value);
            
            if (a_val > b_val) {
                return a;
            } else if (b_val > a_val) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        } else {
            if (a.value > b.value) {
                return a;
            } else if (b.value > a.value) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        }
    }
};

template <typename T>
struct NanArgMinOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const { 
        return ValueIndex<T>(get_max_value<T>(), -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        const bool a_is_nan = is_nan_check(a.value);
        const bool b_is_nan = is_nan_check(b.value);

        if (a_is_nan && b_is_nan) {
            return (a.index < b.index) ? a : b;
        }
        if (a_is_nan) return b;
        if (b_is_nan) return a;

        if constexpr (is_half_float_v<T>) {
            float a_val = static_cast<float>(a.value);
            float b_val = static_cast<float>(b.value);
            
            if (a_val < b_val) {
                return a;
            } else if (b_val < a_val) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        } else {
            if (a.value < b.value) {
                return a;
            } else if (b.value < a.value) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        }
    }
};

template <typename T>
struct NanArgMaxOp {
    using AccumulatorType = ValueIndex<T>;
    
    DEVICE_HOST ValueIndex<T> identity() const {
        T initial_val;
        if constexpr (is_any_float_v<T>) {
            #ifdef __CUDA_ARCH__
                if constexpr (std::is_same_v<T, float>) {
                    initial_val = -CUDART_INF_F;
                } else if constexpr (std::is_same_v<T, double>) {
                    initial_val = -CUDART_INF;
                } else {
                    initial_val = get_lowest_value<T>();
                }
            #else
                if constexpr (std::is_floating_point_v<T>) {
                    initial_val = -std::numeric_limits<T>::infinity();
                } else {
                    initial_val = get_lowest_value<T>();
                }
            #endif
        } else {
            initial_val = get_lowest_value<T>();
        }
        return ValueIndex<T>(initial_val, -1); 
    }

    DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        const bool a_is_nan = is_nan_check(a.value);
        const bool b_is_nan = is_nan_check(b.value);

        if (a_is_nan && b_is_nan) {
            return (a.index < b.index) ? a : b;
        }
        if (a_is_nan) return b;
        if (b_is_nan) return a;

        if constexpr (is_half_float_v<T>) {
            float a_val = static_cast<float>(a.value);
            float b_val = static_cast<float>(b.value);
            
            if (a_val > b_val) {
                return a;
            } else if (b_val > a_val) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        } else {
            if (a.value > b.value) {
                return a;
            } else if (b.value > a.value) {
                return b;
            } else {
                return (a.index < b.index) ? a : b;
            }
        }
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
    NANARGMAX
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

} // namespace detail
} // namespace OwnTensor

#endif // OWNTENSOR_REDUCTION_OPS_H