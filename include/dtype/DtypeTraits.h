#pragma once

#ifndef DTYPE_TRAIT_H
#define DTYPE_TRAIT_H

#include <cstdint>
#include <type_traits>
#include "core/Tensor.h"

// Only include CUDA headers when actually compiling CUDA code
#if defined(WITH_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace OwnTensor
{
    enum class Dtype;

    template <Dtype dt>
    struct dtype_traits 
    {
        using type = void;
        static constexpr size_t size = 0;
        static constexpr const char* name = "invalid";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
    };

    // Integer Types (unchanged)
    template <>
    struct dtype_traits<Dtype::Int16> 
    {
        using type = int16_t;
        static constexpr size_t size = sizeof(int16_t);
        static constexpr const char* name = "int16";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
    };

    template <>
    struct dtype_traits<Dtype::Int32> 
    {
        using type = int32_t;
        static constexpr size_t size = sizeof(int32_t);
        static constexpr const char* name = "int32";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
    };

    template <>
    struct dtype_traits<Dtype::Int64> 
    {
        using type = int64_t;
        static constexpr size_t size = sizeof(int64_t);
        static constexpr const char* name = "int64";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
    };

    // Floating point types
    template <>
    struct dtype_traits<Dtype::Float16> 
    {
        #if defined(WITH_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
        using type = __half;
        static constexpr size_t size = sizeof(__half);
        #else
        using type = float16_t;
        static constexpr size_t size = sizeof(float16_t);
        #endif
        static constexpr const char* name = "fp16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <>
    struct dtype_traits<Dtype::Bfloat16> 
    {   
        #if defined(WITH_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
        using type = __nv_bfloat16;  // FIXED: was __nv_bfloat16_t
        static constexpr size_t size = sizeof(__nv_bfloat16);
        #else
        using type = bfloat16_t;
        static constexpr size_t size = sizeof(bfloat16_t);
        #endif
        static constexpr const char* name = "bf16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <>
    struct dtype_traits<Dtype::Float32> 
    {
        using type = float;
        static constexpr size_t size = sizeof(float);
        static constexpr const char* name = "float32";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    template <>
    struct dtype_traits<Dtype::Float64> 
    {
        using type = double;
        static constexpr size_t size = sizeof(double);
        static constexpr const char* name = "float64";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
    };

    // Utility functions
template<typename T>
constexpr Dtype type_to_dtype()
{
    // Handle all integer type aliases
    if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, short>) {
        return Dtype::Int16;
    }
    else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
        return Dtype::Int32;
    }
    else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, long> || std::is_same_v<T, long long>) {
        return Dtype::Int64;
    }
    // Handle the exact types your library expects
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

    // Type Predicates
    constexpr bool is_float(Dtype dt)
    {
        switch (dt)
        {
        case Dtype::Float16:
        case Dtype::Bfloat16:
        case Dtype::Float32:
        case Dtype::Float64:
            return true;
            break;
        
        default:
            return false;
            break;
        }
    }

    constexpr bool is_int(Dtype dt)
    {
        switch (dt)
        {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
            return true;
            break;
        
        default:
            return false;
            break;
        }
    }

    /**
 * @brief Helper function to get the Dtype name at runtime (since Dtype is a runtime variable).
 */
// inline const char* get_dtype_name(Dtype dtype) {
//     switch (dtype) {
//         case Dtype::Int16:    return "int16";
//         case Dtype::Int32:    return "int32";
//         case Dtype::Int64:    return "int64";
//         case Dtype::Bfloat16: return "bfloat16";
//         case Dtype::Float16:  return "float16";
//         case Dtype::Float32:  return "float32";
//         case Dtype::Float64:  return "float64";
//         default:              return "Unknown";
//     }
// }
inline std::string get_dtype_name(Dtype dtype) {
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
}
#endif // DTYPE_TRAIT_H