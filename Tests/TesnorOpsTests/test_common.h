#pragma once
#include "TensorLib.h"
#include <iostream>
#include <vector>
#include <limits>

using namespace OwnTensor;

// Data types to test - add as needed
constexpr std::array<Dtype, 6> test_dtypes = {
    Dtype::Int64, Dtype::Int32, Dtype::Float64, Dtype::Float32, Dtype::Float16, Dtype::Bfloat16
};

// Device strings
constexpr std::array<const char*, 2> devices = { "cpu", "cuda" };

// Simple data for testing - covers positives, negatives, zeros, max/min, inf/nan for floats
template<typename T>
std::vector<T> get_test_data() {
    return {
        static_cast<T>(1),
        static_cast<T>(-1),
        static_cast<T>(0),
        std::numeric_limits<T>::max(),
        std::numeric_limits<T>::lowest(),
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::infinity()
    };
}
