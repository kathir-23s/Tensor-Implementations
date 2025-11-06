#pragma once

#ifndef DTYPE_H
#define DTYPE_H

namespace OwnTensor {
    // Core dtype enumeration used throughout the library
    enum class Dtype {
        Int16, Int32, Int64,
        Bfloat16, Float16, Float32, Float64,Bool
    };
} // namespace OwnTensor

#endif // DTYPE_H