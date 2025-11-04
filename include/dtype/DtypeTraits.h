// include/dtype/DtypeTraits.h - MERGED VERSION (Compatible with both versions)
#pragma once

#ifndef DTYPE_TRAIT_H
#define DTYPE_TRAIT_H

#include <cstdint>
#include <type_traits>
#include <string>
#include "core/Tensor.h"

// ✅ ALWAYS use custom structs (both CPU and GPU compilation)
#include "dtype/Types.h"

// ═══════════════════════════════════════════════════════════
// DTYPE TRAITS
// ═══════════════════════════════════════════════════════════

namespace OwnTensor {
    enum class Dtype;

    template <Dtype dt>
    struct dtype_traits {
        using type = void;
        static constexpr size_t size = 0;
        static constexpr const char* name = "invalid";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
    };

    // Integer Types
    template <> struct dtype_traits<Dtype::Int16> {
        using type = int16_t;
        static constexpr size_t size = sizeof(int16_t);
        static constexpr const char* name = "int16";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
    };

    template <> struct dtype_traits<Dtype::Int32> {
        using type = int32_t;
        static constexpr size_t size = sizeof(int32_t);
        static constexpr const char* name = "int32";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
    };

    template <> struct dtype_traits<Dtype::Int64> {
        using type = int64_t;
        static constexpr size_t size = sizeof(int64_t);
        static constexpr const char* name = "int64";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
    };

    // ✅ Floating Point Types - ALWAYS use custom structs
    template <> struct dtype_traits<Dtype::Float16> {
        using type = float16_t;  // Custom struct
        static constexpr size_t size = sizeof(float16_t);
        static constexpr const char* name = "fp16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <> struct dtype_traits<Dtype::Bfloat16> {   
        using type = bfloat16_t;  // Custom struct
        static constexpr size_t size = sizeof(bfloat16_t);
        static constexpr const char* name = "bf16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <> struct dtype_traits<Dtype::Float32> {
        using type = float;
        static constexpr size_t size = sizeof(float);
        static constexpr const char* name = "float32";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <> struct dtype_traits<Dtype::Float64> {
        using type = double;
        static constexpr size_t size = sizeof(double);
        static constexpr const char* name = "float64";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    // Add bool specialization
    template<> struct dtype_traits<Dtype::Bool> {
        static constexpr Dtype dtype = Dtype::Bool;
        static constexpr size_t size = sizeof(bool);  // Usually 1 byte
        using type = bool;
        using storage_type = uint8_t;
    };
    // Helper function
    template<typename T>
    bool is_same_type(Dtype dtype) {
        if constexpr (std::is_same_v<T, int32_t>) {
            return dtype == Dtype::Int32;
        } else if constexpr (std::is_same_v<T, float>) {
            return dtype == Dtype::Float32;
        } else if constexpr (std::is_same_v<T, double>) {
            return dtype == Dtype::Float64;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return dtype == Dtype::Int16;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return dtype == Dtype::Int64;
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return dtype == Dtype::Float16; 
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return dtype == Dtype::Bfloat16;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return dtype == Dtype::Bool;
        }
        return false;
    }

    // ═══════════════════════════════════════════════════════════
    // TYPE TO DTYPE CONVERSION
    // ═══════════════════════════════════════════════════════════

    template<typename T>
    constexpr Dtype type_to_dtype() {
        // Integer types
        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, short>) {
            return Dtype::Int16;
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
            return Dtype::Int32;
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, long> || 
                           std::is_same_v<T, long long>) {
            return Dtype::Int64;
        }
        // Floating point types - custom structs
        else if constexpr (std::is_same_v<T, float16_t>) {
            return Dtype::Float16;
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return Dtype::Bfloat16;
        }
        else if constexpr (std::is_same_v<T, float>) {
            return Dtype::Float32;
        }
        else if constexpr (std::is_same_v<T, double>) {
            return Dtype::Float64;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return Dtype::Bool;
        }
        else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // TYPE PREDICATES (MERGED FROM BOTH VERSIONS)
    // ═══════════════════════════════════════════════════════════

    // ✅ Teammate's version (constexpr switch-based)
    constexpr bool is_float(Dtype dt) {
        switch (dt) {
        case Dtype::Float16:
        case Dtype::Bfloat16:
        case Dtype::Float32:
        case Dtype::Float64:
            return true;
        default:
            return false;
        }
    }

    // ✅ Teammate's version (constexpr switch-based)
    constexpr bool is_int(Dtype dt) {
        switch (dt) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
            return true;
        default:
            return false;
        }
    }
    constexpr bool is_bool(Dtype dt) {
        return dt == Dtype::Bool;
    }
    
    // ═══════════════════════════════════════════════════════════
    // DTYPE NAME HELPER (MERGED - Returns std::string like teammate's)
    // ═══════════════════════════════════════════════════════════

    // ✅ Teammate's version returns std::string, yours returned const char*
    // Using std::string for consistency with teammate's code
    inline std::string get_dtype_name(Dtype dtype) {
        switch(dtype) {
            case Dtype::Int16:    return "int16";
            case Dtype::Int32:    return "int32";
            case Dtype::Int64:    return "int64";
            case Dtype::Float16:  return "float16";  // Teammate uses "float16", you used "fp16"
            case Dtype::Bfloat16: return "bfloat16"; // Teammate uses "bfloat16", you used "bf16"
            case Dtype::Float32:  return "float32";
            case Dtype::Float64:  return "float64";
            case Dtype::Bool:     return "bool";
            default:              return "Unknown";  // Teammate uses "Unknown", you used "unknown"
        }
    }

     // Determine element size based on data type
    inline size_t Tensor::dtype_size(Dtype d) {
        switch(d) {
            case Dtype::Int16: return dtype_traits<Dtype::Int16>::size;
            case Dtype::Int32: return dtype_traits<Dtype::Int32>::size;
            case Dtype::Int64: return dtype_traits<Dtype::Int64>::size;
            case Dtype::Bfloat16: return dtype_traits<Dtype::Bfloat16>::size;
            case Dtype::Float16: return dtype_traits<Dtype::Float16>::size;
            case Dtype::Float32: return dtype_traits<Dtype::Float32>::size;
            case Dtype::Float64: return dtype_traits<Dtype::Float64>::size;
            default: throw std::runtime_error("Unsupported data type");
        }
    }

} // namespace OwnTensor

#endif // DTYPE_TRAIT_H