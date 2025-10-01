#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

// ===========================================================
// Portable conversion between Float16/BFloat16 <-> Float32
// ===========================================================

// ---- BFloat16 ----
inline float bfloat16_to_float(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
inline uint16_t float_to_bfloat16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb; // round-to-nearest-even
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

// ---- Float16 (IEEE half precision) ----
inline float float16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t frac = (h & 0x03FFu);
    uint32_t u;

    if (exp == 0) {
        if (frac == 0) {
            u = sign; // +/- 0
        } else {
            float f = static_cast<float>(frac) / 1024.0f;
            f = std::ldexp(f, -14);
            if (sign) f = -f;
            std::memcpy(&u, &f, sizeof(f));
        }
    } else if (exp == 0x1F) {
        u = sign | 0x7F800000u | (frac << 13); // Inf/NaN
    } else {
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
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign); // underflow to 0
        mant |= 0x00800000u; // restore implicit 1
        uint32_t shift = static_cast<uint32_t>(1 - exp);
        uint32_t half_mant = mant >> (shift + 13);
        return static_cast<uint16_t>(sign | half_mant);
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u); // Inf
    } else {
        uint16_t half_exp  = (uint16_t)exp;
        uint32_t half_mant = mant + 0x00001000u; // round
        return static_cast<uint16_t>(sign | (half_exp << 10) | (half_mant >> 13));
    }
}
