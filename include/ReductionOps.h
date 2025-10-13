#pragma once

#ifndef OWNTENSOR_REDUCTION_OPS_H
#define OWNTENSOR_REDUCTION_OPS_H

#include "Types.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <cstdint>

namespace OwnTensor {
namespace detail {

// =================================================================
// HELPER TRAITS
// =================================================================

template <typename T>
constexpr bool is_half_float_v = std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>;

template <typename T>
constexpr bool is_any_float_v = std::is_floating_point_v<T> || is_half_float_v<T>;

// =================================================================
// VALUE-INDEX PAIR FOR ARG REDUCTIONS
// =================================================================

template <typename T>
struct ValueIndex {
    T value;
    int64_t index;

    ValueIndex() : value(T{}), index(-1) {}
    ValueIndex(T val, int64_t idx) : value(val), index(idx) {}

    bool operator>(const ValueIndex<T>& other) const {
        return value > other.value;
    }
    bool operator<(const ValueIndex<T>& other) const {
        return value < other.value;
    }
};

// =================================================================
// HELPER FUNCTIONS
// =================================================================

template <typename T>
T get_lowest_value() {
    if constexpr (std::is_arithmetic_v<T> || is_half_float_v<T>) {
        return std::numeric_limits<T>::lowest();
    }
    throw std::runtime_error("Unsupported type for lowest value");
}

template <typename T>
T get_max_value() {
    if constexpr (std::is_arithmetic_v<T> || is_half_float_v<T>) {
        return std::numeric_limits<T>::max();
    }
    throw std::runtime_error("Unsupported type for max value");
}

template <typename T>
inline bool is_nan_check(T val) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(val);
    } else if constexpr (is_half_float_v<T>) {
        return std::isnan(static_cast<float>(val)); 
    }
    return false; 
}

// =================================================================
// ACCUMULATOR TYPE SELECTOR
// =================================================================

template<typename T>
struct AccumulatorTypeSelector {
    using type = T; // Default: same type
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

// =================================================================
// CORE REDUCTION OPERATIONS
// =================================================================

// --- Sum ---
template <typename T>
struct SumOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { 
        return AccT(0); 
    }
    
    AccT reduce(const AccT& a, const AccT& b) const { 
        return a + b; 
    }
};

// --- Product ---
template <typename T>
struct ProductOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { 
        return AccT(1); 
    }
    
    AccT reduce(const AccT& a, const AccT& b) const { 
        return a * b; 
    }
};

// --- Min ---
template <typename T>
struct MinOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { 
        // CRITICAL FIX: Handle signed vs unsigned integers properly
        if constexpr (std::is_integral_v<T>) {
            // For signed integers, return max of the ORIGINAL type T (not AccT)
            // Then cast to AccT (int64_t) to avoid overflow warnings
            if constexpr (std::is_signed_v<T>) {
                return static_cast<AccT>(std::numeric_limits<T>::max());
            } else {
                // For unsigned integers, return max of T directly
                return static_cast<AccT>(std::numeric_limits<T>::max());
            }
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        } else {
            return std::numeric_limits<AccT>::max();
        }
    }
    
    AccT reduce(const AccT& a, const AccT& b) const { 
        return std::min(a, b); 
    }
};

// --- Max ---
template <typename T>
struct MaxOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { 
        // CRITICAL FIX: Handle signed vs unsigned integers properly
        if constexpr (std::is_integral_v<T>) {
            // For signed integers, return lowest of the ORIGINAL type T
            if constexpr (std::is_signed_v<T>) {
                return static_cast<AccT>(std::numeric_limits<T>::lowest());
            } else {
                // For unsigned integers, lowest is 0
                return static_cast<AccT>(std::numeric_limits<T>::lowest());
            }
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        } else {
            return std::numeric_limits<AccT>::lowest();
        }
    }
    
    AccT reduce(const AccT& a, const AccT& b) const {
        // NaN propagation for floats
        if constexpr (std::is_floating_point_v<AccT>) {
            if (std::isnan(a)) return a;
            if (std::isnan(b)) return b;
        }
        return std::max(a, b);
    }
};

// =================================================================
// NaN-AWARE OPERATIONS
// =================================================================

template <typename T>
struct NanSumOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { return AccT(0); }
    
    AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT>) {
            if (std::isnan(a)) return b;
            if (std::isnan(b)) return a;
        }
        return a + b;
    }
};

template <typename T>
struct NanProductOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { return AccT(1); }
    
    AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT>) {
            if (std::isnan(a)) return b;
            if (std::isnan(b)) return a;
        }
        return a * b;
    }
};

template <typename T>
struct NanMinOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { 
        // CRITICAL FIX: Handle signed vs unsigned integers properly
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                return static_cast<AccT>(std::numeric_limits<T>::max());
            } else {
                return static_cast<AccT>(std::numeric_limits<T>::max());
            }
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_max_value<T>());
        } else {
            return std::numeric_limits<AccT>::max();
        }
    }
    
    AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT>) {
            if (std::isnan(a)) return b;
            if (std::isnan(b)) return a;
        }
        return std::min(a, b);
    }
};

template <typename T>
struct NanMaxOp {
    using AccT = AccumulatorType<T>;
    
    AccT identity() const { 
        // CRITICAL FIX: Handle signed vs unsigned integers properly
        if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                return static_cast<AccT>(std::numeric_limits<T>::lowest());
            } else {
                return static_cast<AccT>(std::numeric_limits<T>::lowest());
            }
        } else if constexpr (is_half_float_v<T>) {
            return static_cast<AccT>(get_lowest_value<T>());
        } else {
            return std::numeric_limits<AccT>::lowest();
        }
    }
    
    AccT reduce(const AccT& a, const AccT& b) const {
        if constexpr (std::is_floating_point_v<AccT>) {
            if (std::isnan(a)) return b;
            if (std::isnan(b)) return a;
        }
        return std::max(a, b);
    }
};

// =================================================================
// INDEX REDUCTIONS
// =================================================================

template <typename T>
struct ArgMinOp {
    using AccumulatorType = ValueIndex<T>;
    
    ValueIndex<T> identity() const { 
        return ValueIndex<T>(get_max_value<T>(), -1); 
    }

    ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        if (a.value < b.value) {
            return a;
        } else if (b.value < a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
    }
};

template <typename T>
struct ArgMaxOp {
    using AccumulatorType = ValueIndex<T>;
    
    ValueIndex<T> identity() const {
        T initial_val;
        if constexpr (is_any_float_v<T>) {
             // Note: using negative infinity for floats/half-floats
             initial_val = -std::numeric_limits<T>::infinity();
        } else {
             initial_val = get_lowest_value<T>();
        }
        return ValueIndex<T>(initial_val, -1); 
    }

    ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        if (a.value > b.value) {
            return a;
        } else if (b.value > a.value) {
            return b;
        } else {
            // Tie-breaking: return the smaller index
            return (a.index < b.index) ? a : b;
        }
    }
};

template <typename T>
struct NanArgMinOp {
    using AccumulatorType = ValueIndex<T>;
    
    ValueIndex<T> identity() const { 
        return ValueIndex<T>(get_max_value<T>(), -1); 
    }

    ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        const bool a_is_nan = is_nan_check(a.value);
        const bool b_is_nan = is_nan_check(b.value);

        if (a_is_nan && b_is_nan) {
            return (a.index < b.index) ? a : b;
        }
        if (a_is_nan) return b;
        if (b_is_nan) return a;

        if (a.value < b.value) {
            return a;
        } else if (b.value < a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
    }
};

template <typename T>
struct NanArgMaxOp {
    using AccumulatorType = ValueIndex<T>;
    
    ValueIndex<T> identity() const {
        T initial_val;
        if constexpr (is_any_float_v<T>) {
             initial_val = -std::numeric_limits<T>::infinity();
        } else {
             initial_val = get_lowest_value<T>();
        }
        return ValueIndex<T>(initial_val, -1); 
    }

    ValueIndex<T> reduce(const ValueIndex<T>& a, const ValueIndex<T>& b) const {
        const bool a_is_nan = is_nan_check(a.value);
        const bool b_is_nan = is_nan_check(b.value);

        if (a_is_nan && b_is_nan) {
            return (a.index < b.index) ? a : b;
        }
        if (a_is_nan) return b;
        if (b_is_nan) return a;

        if (a.value > b.value) {
            return a;
        } else if (b.value > a.value) {
            return b;
        } else {
            return (a.index < b.index) ? a : b;
        }
    }
};

} // namespace detail
} // namespace OwnTensor

#endif // OWNTENSOR_REDUCTION_OPS_H