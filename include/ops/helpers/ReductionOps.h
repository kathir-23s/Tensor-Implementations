// include/ops/helpers/ReductionOps.h - REVERTED TO CUSTOM STRUCTS
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

// ✅ ALWAYS use custom structs (both CPU and GPU)
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
// DEVICE-SAFE HELPER FUNCTIONS (NO std::numeric_limits)
// ═══════════════════════════════════════════════════════════

template <typename T>
DEVICE_HOST constexpr T get_lowest_value() {
    if constexpr (std::is_same_v<T, float16_t>) {
        return T(-65504.0f);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return T(-3.38953e38f);
    } else if constexpr (std::is_same_v<T, float>) {
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
    } else if constexpr (std::is_same_v<T, float>) {
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
    if constexpr (std::is_floating_point_v<T>) {
        #ifdef __CUDA_ARCH__
            return isnan(val);
        #else
            return std::isnan(val);
        #endif
    } else if constexpr (is_half_float_v<T>) {
        // Convert to float and check
        float f_val = static_cast<float>(val);
        #ifdef __CUDA_ARCH__
            return isnan(f_val);
        #else
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

template<> struct AccumulatorTypeSelector<int16_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int32_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<int64_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint16_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint32_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<uint64_t> { using type = int64_t; };
template<> struct AccumulatorTypeSelector<float16_t> { using type = float; };
template<> struct AccumulatorTypeSelector<bfloat16_t> { using type = float; };

template<typename T>
using AccumulatorType = typename AccumulatorTypeSelector<T>::type;

// ═══════════════════════════════════════════════════════════
// CORE REDUCTION OPERATIONS (NO INTRINSICS)
// ═══════════════════════════════════════════════════════════

template <typename T>
struct SumOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        return AccT(0); 
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        return a + b;
    }
};

template <typename T>
struct ProductOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { 
        return AccT(1); 
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        return a * b;
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
        } else {
            return get_max_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const { 
        return (a < b) ? a : b;
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
        } else {
            return get_lowest_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        return (a > b) ? a : b;
    }
};

// ═══════════════════════════════════════════════════════════
// NaN-AWARE OPERATIONS
// ═══════════════════════════════════════════════════════════

template <typename T>
struct NanSumOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { return AccT(0); }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return a + b;
    }
};

template <typename T>
struct NanProductOp {
    using AccT = AccumulatorType<T>;
    
    DEVICE_HOST AccT identity() const { return AccT(1); }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return a * b;
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
        } else {
            return get_max_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return (a < b) ? a : b;
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
        } else {
            return get_lowest_value<AccT>();
        }
    }
    
    DEVICE_HOST AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
            if (is_nan_check(a)) return b;
            if (is_nan_check(b)) return a;
        }
        return (a > b) ? a : b;
    }
};

// ═══════════════════════════════════════════════════════════
// INDEX REDUCTIONS (ArgMin/ArgMax)
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
                initial_val = get_lowest_value<T>();
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
                initial_val = get_lowest_value<T>();
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