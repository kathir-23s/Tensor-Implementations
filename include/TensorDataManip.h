#pragma once

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "Tensor.h"
#include "Types.h"
#include "device/DeviceTransfer.h"  // Add this include
#include <iostream>
#include <cstring>

namespace OwnTensor
{
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

        // Use device-aware copy instead of memcpy
        device::copy_memory(data_ptr_.get(), device_.device,
                           source_data, Device::CPU,
                           count * sizeof(T));
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

        // For fill operations, we need to handle device properly
        if (device_.is_cpu()) {
            T* data = reinterpret_cast<T*>(data_ptr_.get());
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = value;
            }
        } else {
            // For GPU, create a temporary CPU buffer and transfer
            std::vector<T> temp_data(numel(), value);
            set_data(temp_data);
        }
    }
    template <typename T>
    inline void Tensor::set_data(std::initializer_list<T> values) {
        // if (dtype_ != Dtype::Float32) {
        //     throw std::runtime_error("Initializer list only supports Float32");
        // }
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
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return dtype == Dtype::Float16; 
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return dtype == Dtype::Bfloat16;
        }
        return false;
    }
}
#endif