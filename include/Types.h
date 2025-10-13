#pragma once

#include <cstdint>
#include <limits>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace OwnTensor {

// ==================================================================================
// SECTION 1: LOW-LEVEL CONVERSION FUNCTIONS (From fp16_bf16_convert.h)
// ==================================================================================
// These are the mathematically correct, portable conversion routines.

namespace detail {

// ---- BFloat16 Conversions (Google Brain Float) ----
inline float bfloat16_to_float(uint16_t b) {
    // BFloat16: Sign(1) + Exp(8) + Mantissa(7) -> Just shift left 16 bits
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t float_to_bfloat16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round-to-nearest-even (RNE) logic
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

// ---- Float16 Conversions (IEEE 754 Half-Precision) ----
inline float float16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t frac = (h & 0x03FFu);
    uint32_t u;

    if (exp == 0) {
        if (frac == 0) {
            // Zero (preserve sign)
            u = sign;
        } else {
            // Denormal number
            float f = static_cast<float>(frac) / 1024.0f;
            f = std::ldexp(f, -14);
            if (sign) f = -f;
            std::memcpy(&u, &f, sizeof(f));
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        u = sign | 0x7F800000u | (frac << 13);
    } else {
        // Normal number
        uint32_t exp32 = exp + (127 - 15);
        u = sign | (exp32 << 23) | (frac << 13);
    }
    
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t float_to_float16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp_32 = (x >> 23) & 0xFF;
    uint32_t mant_32 = x & 0x007FFFFFu;

    // 1. Handle NaN and Infinity
    if (exp_32 == 0xFF) {
        if (mant_32 == 0) {
            // Infinity
            return static_cast<uint16_t>(sign | 0x7C00u);
        } else {
            // NaN
            uint32_t qnan_mant = mant_32 >> 13;
            if (qnan_mant == 0) {
                 qnan_mant = 1; 
            }
            return static_cast<uint16_t>(sign | 0x7C00u | qnan_mant);
        }
    }

    // 2. Handle Normal/Denormal conversion
    int32_t exp_16  = static_cast<int32_t>(exp_32) - 127 + 15;

    if (exp_16 <= 0) {
        // Underflow handling
        if (exp_16 < -10) {
            return static_cast<uint16_t>(sign);
        }
        // Denormal conversion
        mant_32 |= 0x00800000u;
        uint32_t shift = static_cast<uint32_t>(1 - exp_16);
        uint32_t half_mant = mant_32 >> (shift + 13);
        return static_cast<uint16_t>(sign | half_mant);
    } else if (exp_16 >= 31) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00u);
    } else {
        // Normal conversion with rounding
        uint16_t half_exp  = static_cast<uint16_t>(exp_16);
        uint32_t half_mant = mant_32 + 0x00001000u;
        return static_cast<uint16_t>(sign | (half_exp << 10) | (half_mant >> 13));
    }
}

} // namespace detail

// ==================================================================================
// SECTION 2: C++ TYPE WRAPPERS (16-bit Floating Point Types)
// ==================================================================================

/**
 * @brief BFloat16 (Brain Floating Point 16)
 * Format: Sign(1) + Exponent(8) + Mantissa(7)
 * Range: Same as Float32, reduced precision
 * Use case: Deep learning, faster than FP32, wider range than FP16
 */
struct bfloat16_t {
    uint16_t raw_bits;

    // ---- Constructors ----
    bfloat16_t() : raw_bits(0) {}
    
    explicit bfloat16_t(float val) {
        raw_bits = detail::float_to_bfloat16(val);
    }

    bfloat16_t(const bfloat16_t& other) : raw_bits(other.raw_bits) {}

    template <typename U, typename = std::enable_if_t<
        std::is_arithmetic_v<U> && !std::is_same_v<std::decay_t<U>, float>
    >>
    explicit bfloat16_t(U val) {
        raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
    }

    // ---- Conversion ----
    operator float() const {
        return detail::bfloat16_to_float(raw_bits);
    }

    // ---- Assignment Operators ----
    bfloat16_t& operator=(float val) {
        raw_bits = detail::float_to_bfloat16(val);
        return *this;
    }
    
    bfloat16_t& operator=(const bfloat16_t& other) {
        raw_bits = other.raw_bits;
        return *this;
    }

    template <typename U, typename = std::enable_if_t<
        std::is_arithmetic_v<U> && !std::is_same_v<std::decay_t<U>, float>
    >>
    bfloat16_t& operator=(U val) {
        raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
        return *this;
    }

    // ---- Comparison Operators ----
    bool operator>(const bfloat16_t& other) const {
        return static_cast<float>(*this) > static_cast<float>(other);
    }
    bool operator<(const bfloat16_t& other) const {
        return static_cast<float>(*this) < static_cast<float>(other);
    }
    bool operator>=(const bfloat16_t& other) const {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }
    bool operator<=(const bfloat16_t& other) const {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }
    bool operator==(const bfloat16_t& other) const {
        return raw_bits == other.raw_bits;
    }
    bool operator!=(const bfloat16_t& other) const {
        return raw_bits != other.raw_bits;
    }

    // ---- Arithmetic Operators ----
    bfloat16_t operator+(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) + static_cast<float>(other));
    }
    
    bfloat16_t operator-(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) - static_cast<float>(other));
    }
    
    bfloat16_t operator*(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) * static_cast<float>(other));
    }
    
    bfloat16_t operator/(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) / static_cast<float>(other));
    }

    // ---- Compound Assignment Operators ----
    bfloat16_t& operator+=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) + static_cast<float>(other));
        return *this;
    }
    
    bfloat16_t& operator-=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) - static_cast<float>(other));
        return *this;
    }
    
    bfloat16_t& operator*=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) * static_cast<float>(other));
        return *this;
    }
    
    bfloat16_t& operator/=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) / static_cast<float>(other));
        return *this;
    }
};

/**
 * @brief Float16 (IEEE 754 Half Precision)
 * Format: Sign(1) + Exponent(5) + Mantissa(10)
 * Range: ±65504, limited but higher precision than BF16
 * Use case: Graphics, mobile AI, memory-constrained scenarios
 */
struct float16_t {
    uint16_t raw_bits;

    // ---- Constructors ----
    float16_t() : raw_bits(0) {}
    
    explicit float16_t(float val) {
        raw_bits = detail::float_to_float16(val);
    }

    float16_t(const float16_t& other) : raw_bits(other.raw_bits) {}

    template <typename U, typename = std::enable_if_t<
        std::is_arithmetic_v<U> && !std::is_same_v<std::decay_t<U>, float>
    >>
    explicit float16_t(U val) {
        raw_bits = detail::float_to_float16(static_cast<float>(val));
    }

    // ---- Conversion ----
    operator float() const {
        return detail::float16_to_float(raw_bits);
    }

    // ---- Assignment Operators ----
    float16_t& operator=(float val) {
        raw_bits = detail::float_to_float16(val);
        return *this;
    }
    
    float16_t& operator=(const float16_t& other) {
        raw_bits = other.raw_bits;
        return *this;
    }

    template <typename U, typename = std::enable_if_t<
        std::is_arithmetic_v<U> && !std::is_same_v<std::decay_t<U>, float>
    >>
    float16_t& operator=(U val) {
        raw_bits = detail::float_to_float16(static_cast<float>(val));
        return *this;
    }

    // ---- Comparison Operators ----
    bool operator>(const float16_t& other) const {
        return static_cast<float>(*this) > static_cast<float>(other);
    }
    bool operator<(const float16_t& other) const {
        return static_cast<float>(*this) < static_cast<float>(other);
    }
    bool operator>=(const float16_t& other) const {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }
    bool operator<=(const float16_t& other) const {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }
    bool operator==(const float16_t& other) const {
        return raw_bits == other.raw_bits;
    }
    bool operator!=(const float16_t& other) const {
        return raw_bits != other.raw_bits;
    }

    // ---- Arithmetic Operators ----
    float16_t operator+(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) + static_cast<float>(other));
    }
    
    float16_t operator-(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) - static_cast<float>(other));
    }
    
    float16_t operator*(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) * static_cast<float>(other));
    }
    
    float16_t operator/(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) / static_cast<float>(other));
    }

    // ---- Compound Assignment Operators ----
    float16_t& operator+=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) + static_cast<float>(other));
        return *this;
    }
    
    float16_t& operator-=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) - static_cast<float>(other));
        return *this;
    }
    
    float16_t& operator*=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) * static_cast<float>(other));
        return *this;
    }
    
    float16_t& operator/=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) / static_cast<float>(other));
        return *this;
    }
};

// ==================================================================================
// SECTION 3: TYPE TRAITS AND EXTENSIONS (Future-Ready)
// ==================================================================================

/**
 * @brief Type trait to check if a type is a half-precision float.
 * Extensible for future types.
 */
template <typename T>
struct is_half_float : std::false_type {};

template <>
struct is_half_float<bfloat16_t> : std::true_type {};

template <>
struct is_half_float<float16_t> : std::true_type {};

template <typename T>
inline constexpr bool is_half_float_v = is_half_float<T>::value;

/**
 * @brief Combined type trait: checks if T is any floating point (standard or custom).
 */
template <typename T>
inline constexpr bool is_any_float_v = std::is_floating_point_v<T> || is_half_float_v<T>;

// ==================================================================================
// FUTURE TYPE PLACEHOLDERS (To be implemented later)
// ==================================================================================

// Uncomment and implement when needed:
/*
struct int8_t_custom { int8_t value; };
struct uint8_t_custom { uint8_t value; };
struct float8_e4m3_t { uint8_t raw_bits; }; // FP8 E4M3 format
struct float8_e5m2_t { uint8_t raw_bits; }; // FP8 E5M2 format
*/

} // namespace OwnTensor

// ==================================================================================
// SECTION 4: std::numeric_limits SPECIALIZATIONS (Must be in std namespace)
// ==================================================================================

namespace std {

/**
 * @brief Helper template for numeric_limits of 16-bit float types.
 */
template <typename T>
struct numeric_limits_fp16_helper {
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    
    static T lowest() noexcept;
    static T max() noexcept;
    static T min() noexcept { return T(1.0e-8f); }
    static T epsilon() noexcept { return T(0.0001f); }
    static T infinity() noexcept { return T(std::numeric_limits<float>::infinity()); }
    static T quiet_NaN() noexcept { return T(std::numeric_limits<float>::quiet_NaN()); }
};

// ---- BFloat16 Limits ----
template <>
struct numeric_limits<OwnTensor::bfloat16_t> : public numeric_limits_fp16_helper<OwnTensor::bfloat16_t> {};

template <>
inline OwnTensor::bfloat16_t numeric_limits_fp16_helper<OwnTensor::bfloat16_t>::lowest() noexcept {
    return OwnTensor::bfloat16_t(-3.38953e38f);
}

template <>
inline OwnTensor::bfloat16_t numeric_limits_fp16_helper<OwnTensor::bfloat16_t>::max() noexcept {
    return OwnTensor::bfloat16_t(3.38953e38f);
}

// ---- Float16 Limits ----
template <>
struct numeric_limits<OwnTensor::float16_t> : public numeric_limits_fp16_helper<OwnTensor::float16_t> {};

template <>
inline OwnTensor::float16_t numeric_limits_fp16_helper<OwnTensor::float16_t>::lowest() noexcept {
    return OwnTensor::float16_t(-65504.0f);
}

template <>
inline OwnTensor::float16_t numeric_limits_fp16_helper<OwnTensor::float16_t>::max() noexcept {
    return OwnTensor::float16_t(65504.0f);
}

} // namespace std

// ==================================================================================
// SECTION 5: DOCUMENTATION AND USAGE GUIDE
// ==================================================================================

/*
 * ==================================================================================
 * USAGE EXAMPLES:
 * ==================================================================================
 * 
 * // 1. Basic creation and conversion
 * OwnTensor::bfloat16_t bf16_val(3.14f);
 * OwnTensor::float16_t fp16_val(2.71f);
 * float f = static_cast<float>(bf16_val);
 * 
 * // 2. Arithmetic operations
 * auto result = bf16_val + bf16_val;
 * bf16_val *= 2.0f;
 * 
 * // 3. Template usage (automatically works with reduction kernels)
 * template <typename T>
 * T compute(T a, T b) { return a * b + T(1.0f); }
 * auto res = compute(bf16_val, OwnTensor::bfloat16_t(5.0f));
 * 
 * // 4. Type traits
 * static_assert(OwnTensor::is_half_float_v<OwnTensor::bfloat16_t>);
 * static_assert(OwnTensor::is_any_float_v<float>);
 * 
 * // 5. Numeric limits
 * auto max_bf16 = std::numeric_limits<OwnTensor::bfloat16_t>::max();
 * auto inf_fp16 = std::numeric_limits<OwnTensor::float16_t>::infinity();
 * 
 * ==================================================================================
 */