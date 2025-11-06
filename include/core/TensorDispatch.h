#pragma once

#ifndef TENSOR_DISPATCH_H
#define TENSOR_DISPATCH_H

// ✅ CRITICAL: Include Dtype.h first for enum definition
#include "dtype/Dtype.h"
#include <stdexcept>
#include <type_traits>
#include <cstdint>

// ✅ Forward declare the Tensor class (avoid circular dependency)
namespace OwnTensor {
    class Tensor;
}

// ✅ CRITICAL FIX: Manually specify the type mapping WITHOUT dtype_traits
// This avoids the circular dependency and NVCC template resolution issues
namespace OwnTensor {

// ✅ Simple type resolver that works in both CPU and CUDA contexts
template<Dtype dt> struct DtypeToType;

// Integer types
template<> struct DtypeToType<Dtype::Int16>  { using type = int16_t; };
template<> struct DtypeToType<Dtype::Int32>  { using type = int32_t; };
template<> struct DtypeToType<Dtype::Int64>  { using type = int64_t; };

// Standard floating point
template<> struct DtypeToType<Dtype::Float32> { using type = float; };
template<> struct DtypeToType<Dtype::Float64> { using type = double; };

//Boolean type
template<> struct DtypeToType<Dtype::Bool> { using type = bool;};

// ✅ Half precision types - resolve based on compilation context
#ifdef __CUDACC__
    // GPU compilation - use native CUDA types
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    template<> struct DtypeToType<Dtype::Float16>  { using type = __half; };
    template<> struct DtypeToType<Dtype::Bfloat16> { using type = __nv_bfloat16; };
#else
    // CPU compilation - use custom types
    #include "dtype/Types.h"
    template<> struct DtypeToType<Dtype::Float16>  { using type = float16_t; };
    template<> struct DtypeToType<Dtype::Bfloat16> { using type = bfloat16_t; };
#endif

// ✅ Runtime dispatcher using the simple type resolver
template<typename Func>
static auto dispatch_by_dtype(Dtype dtype, Func&& f) {
    switch(dtype) {
        case Dtype::Int16:    return f(typename DtypeToType<Dtype::Int16>::type{});
        case Dtype::Int32:    return f(typename DtypeToType<Dtype::Int32>::type{});
        case Dtype::Int64:    return f(typename DtypeToType<Dtype::Int64>::type{});
        case Dtype::Float32:  return f(typename DtypeToType<Dtype::Float32>::type{});
        case Dtype::Float64:  return f(typename DtypeToType<Dtype::Float64>::type{});
        case Dtype::Bfloat16: return f(typename DtypeToType<Dtype::Bfloat16>::type{});
        case Dtype::Float16:  return f(typename DtypeToType<Dtype::Float16>::type{});
        case Dtype::Bool:   return f(typename DtypeToType<Dtype::Bool>::type{});
        default:
            throw std::runtime_error("Unsupported Dtype");
    }
}

} // namespace OwnTensor

#endif