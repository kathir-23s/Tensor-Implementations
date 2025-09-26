#pragma once

#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

#include "../include/Tensor.h"
#include <iostream>
#include <cstring>


template <typename T>
inline void Tensor::set_data(const T* source_data, size_t count)
{
    if (count != numel())
    {
        throw std::runtime_error("Data size does not match tensor size");
    }

    if (sizeof(T) != dtype_size(dtype_))
    {
        throw std::runtime_error("Data type size mismatch");
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

#endif