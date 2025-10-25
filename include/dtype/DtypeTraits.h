#pragma once

#ifndef DTYPE_TRAIT_H
#define DTYPE_TRAIT_H

#include <cstdint>
#include <type_traits>
#include "Dtype.h"
// ═══════════════════════════════════════════════════════════
// TYPE RESOLUTION: CPU vs GPU
// ═══════════════════════════════════════════════════════════

#ifdef __CUDACC__
    // ✅ GPU COMPILATION (.cu files)
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    
    namespace OwnTensor {
        // Map our API names to native CUDA types
        using float16_t = __half;
        using bfloat16_t = __nv_bfloat16;
    }
#else
    // ✅ CPU COMPILATION (.cpp files)
    #include "Types.h"  // Uses custom structs
#endif




// ═══════════════════════════════════════════════════════════
// DTYPE TRAITS - NOW ALWAYS DEFINED (both CPU and GPU)
// ═══════════════════════════════════════════════════════════

namespace OwnTensor {

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

    // Floating Point Types (same for CPU/GPU thanks to type aliases)
    template <> struct dtype_traits<Dtype::Float16> {
        using type = float16_t;  // ✅ Resolves to __half on GPU, custom on CPU
        static constexpr size_t size = sizeof(float16_t);
        static constexpr const char* name = "fp16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <> struct dtype_traits<Dtype::Bfloat16> {   
        using type = bfloat16_t;  // ✅ Resolves to __nv_bfloat16 on GPU, custom on CPU
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

    // ═══════════════════════════════════════════════════════════
    // HELPER FUNCTIONS
    // ═══════════════════════════════════════════════════════════

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
        return false;
    }

    template<typename T>
    constexpr Dtype type_to_dtype() {
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
        else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type");
        }
    }

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

    inline const char* get_dtype_name(Dtype dtype) {
        switch (dtype) {
            case Dtype::Int16:    return "int16";
            case Dtype::Int32:    return "int32";
            case Dtype::Int64:    return "int64";
            case Dtype::Bfloat16: return "bfloat16";
            case Dtype::Float16:  return "float16";
            case Dtype::Float32:  return "float32";
            case Dtype::Float64:  return "float64";
            default:              return "Unknown";
        }
    }

} // namespace OwnTensor

#endif // DTYPE_TRAIT_H