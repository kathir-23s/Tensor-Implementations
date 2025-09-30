#pragma once

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "../include/Tensor.h"
#include <iostream>
#include <cstring>


// Forward declaration for is_same_type
template<typename T>
bool is_same_type(Dtype dtype);

template <typename T>
inline void Tensor::set_data(const T* source_data, size_t count)
{
    if (count != numel())
    {
        throw std::runtime_error("Data size does not match tensor size");
    }


    if (!is_same_type<T>(dtype_))
    {
        throw std::runtime_error("Datatype mismatch");
    }

    std::memcpy(data_ptr_.get(), source_data, count * sizeof(T));
}

template <typename T>
inline void Tensor::set_data(const std::vector<T>& source_data)
{
    set_data(source_data.data(), source_data.size());
}

template <typename T>
inline void Tensor::fill(T value)
{
    if (sizeof(T) != dtype_size(dtype_))
    {
        throw std::runtime_error("Fill value type mismatch");
    }

    T* data = reinterpret_cast<T*>(data_ptr_.get());
    for (size_t i = 0; i < numel(); ++i) {
        data[i] = value;
    }
}

inline void Tensor::set_data(std::initializer_list<float> values) {
    if (dtype_ != Dtype::Float32) {
        throw std::runtime_error("Initializer list only supports Float32");
    }
    set_data(values.begin(), values.size());
}




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
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return dtype == Dtype::Float16; 
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return dtype == Dtype::Bfloat16;
    }
    return false;
}

#endif



