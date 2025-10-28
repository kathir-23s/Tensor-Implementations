// #pragma once

// #ifndef TENSOR_UTILS_H
// #define TENSOR_UTILS_H

// #include "core/Tensor.h"
// #include "dtype/Types.h"
// #include "device/DeviceTransfer.h"  // Add this include
// #include <iostream>
// #include <cstring>
// #include "dtype/DtypeTraits.h" // For is_same_type

// namespace OwnTensor
// {
    
// // Forward declaration for is_same_type
//     template<typename T>
//     bool is_same_type(Dtype dtype);
    
//     template <typename T>
//     inline void Tensor::set_data(const T* source_data, size_t count)
//     {
//         if (count != numel())
//         {
//             throw std::runtime_error("Data size does not match tensor size");
//         }

//         if (!is_same_type<T>(dtype_))
//         {
//             throw std::runtime_error("Datatype mismatch");
//         }

//         // Use device-aware copy instead of memcpy
//         device::copy_memory(data_ptr_.get(), device_.device,
//                            source_data, Device::CPU,
//                            count * sizeof(T));
//     }

//     template <typename T>
//     inline void Tensor::set_data(const std::vector<T>& source_data)
//     {
//         set_data(source_data.data(), source_data.size());
//     }

//     template <typename T>
//     inline void Tensor::fill(T value)
//     {
//         if (sizeof(T) != dtype_size(dtype_))
//         {
//             throw std::runtime_error("Fill value type mismatch");
//         }

//         // For fill operations, we need to handle device properly
//         if (device_.is_cpu()) {
//             T* data = reinterpret_cast<T*>(data_ptr_.get());
//             for (size_t i = 0; i < numel(); ++i) {
//                 data[i] = value;
//             }
//         } else {
//             // For GPU, create a temporary CPU buffer and transfer
//             std::vector<T> temp_data(numel(), value);
//             set_data(temp_data);
//         }
//     }
//     template <typename T>
//     inline void Tensor::set_data(std::initializer_list<T> values) {
//         // if (dtype_ != Dtype::Float32) {
//         //     throw std::runtime_error("Initializer list only supports Float32");
//         // } (commented this,so that it supports all other dtypes defined too.)
//         set_data(values.begin(), values.size());
//     }

    
// }
// #endif
#pragma once

#ifndef TENSOR_DATAMANIP_H
#define TENSOR_DATAMANIP_H

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "device/DeviceTransfer.h" 
#include <iostream>
#include <cstring>
#include <vector> // Required for temporary vector in specialization
#include "dtype/DtypeTraits.h" // For is_same_type

namespace OwnTensor {
// Forward declaration for is_same_type
    template<typename T>
    bool is_same_type(Dtype dtype);
    
    // =========================================================================
    // GENERIC IMPLEMENTATIONS (Used for standard types: float, double, int32, etc.)
    // =========================================================================
    
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

        // Use device-aware copy for standard types
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

        // For fill operations, we handle device properly
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
        set_data(values.begin(), values.size());
    }


    // =========================================================================
    // SPECIALIZED IMPLEMENTATIONS (CRITICAL FIX for custom 16-bit types)
    // =========================================================================

    // --- Specialization for float16_t ---
    template <>
    inline void Tensor::set_data<float16_t>(const float16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<float16_t>(dtype_)) {
            throw std::runtime_error("Datatype mismatch");
        }

        // FIX: Extract the raw 16-bit data into a contiguous array.
        // We cannot rely on the pointer arithmetic of the custom struct directly.
        std::vector<uint16_t> raw_data(count);
        for (size_t i = 0; i < count; ++i) {
            raw_data[i] = source_data[i].raw_bits;
        }

        // Copy the raw 16-bit integers (uint16_t) to the tensor's memory
        // This ensures the memory layout is correct (2 bytes per element).
        device::copy_memory(data_ptr_.get(), device_.device,
                           raw_data.data(), Device::CPU,
                           count * sizeof(uint16_t));
    }
    
    // Specialization for vector<float16_t> which delegates to the const T* version
    template <>
    inline void Tensor::set_data<float16_t>(const std::vector<float16_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }
    
    // --- Specialization for bfloat16_t ---
    template <>
    inline void Tensor::set_data<bfloat16_t>(const bfloat16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bfloat16_t>(dtype_)) {
            throw std::runtime_error("Datatype mismatch");
        }

        // FIX: Extract the raw 16-bit data into a temporary contiguous buffer.
        std::vector<uint16_t> raw_data(count);
        for (size_t i = 0; i < count; ++i) {
            raw_data[i] = source_data[i].raw_bits;
        }

        // Copy the raw 16-bit integers (uint16_t) to the tensor's memory
        device::copy_memory(data_ptr_.get(), device_.device,
                           raw_data.data(), Device::CPU,
                           count * sizeof(uint16_t));
    }
    
    // Specialization for vector<bfloat16_t> which delegates to the const T* version
    template <>
    inline void Tensor::set_data<bfloat16_t>(const std::vector<bfloat16_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }
    
}
#endif // TENSOR_UTILS_H