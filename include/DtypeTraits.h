#pragma once

#ifndef DTYPE_TRAIT_H
#define DTYPE_TRAIT_H

#include <cstdint>
#include <type_traits>
#include "Tensor.h"

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

// Integer Types
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
    // No native float16 type in standard C++, so use uint16_t as a placeholder
    // Need to look into ways we can make this work
    // TO DO: !!!
    using type = uint16_t;
    static constexpr size_t size = sizeof(uint16_t);
    static constexpr const char* name = "fp16";
    static constexpr bool is_floating_point = true;
    static constexpr bool is_integral = false;
};

template <>
struct dtype_traits<Dtype::Bfloat16> 
{   
    // No native float16 type in standard C++, so use uint16_t as a placeholder
    // Need to look into ways we can make this work
    // TO DO: !!!
    using type = uint16_t;
    static constexpr size_t size = sizeof(uint16_t);
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
    if constexpr (std::is_same_v<T, int16_t>) return Dtype::Int16;
    if constexpr (std::is_same_v<T, int32_t>) return Dtype::Int32;
    if constexpr (std::is_same_v<T, int64_t>) return Dtype::Int64;
    if constexpr (std::is_same_v<T, uint16_t>) return Dtype::Float16;
    if constexpr (std::is_same_v<T, uint16_t>) return Dtype::Bfloat16;
    if constexpr (std::is_same_v<T, float>) return Dtype::Float32;
    if constexpr (std::is_same_v<T, double>) return Dtype::Float64;
    static_assert(!std::is_same_v<T, T>, "Unsupported type");  // Force error
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

#endif // DTYPE_TRAIT_H