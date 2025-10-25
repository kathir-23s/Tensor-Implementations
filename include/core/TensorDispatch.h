#pragma once

#ifndef TENSOR_DISPATCH_H
#define TENSOR_DISPATCH_H

#include "dtype/DtypeTraits.h"  // ✅ Include BEFORE Tensor.h
#include "core/Tensor.h"
#include <stdexcept>
#include <type_traits>

namespace OwnTensor {

template<typename Func>
static auto dispatch_by_dtype(Dtype dtype, Func&& f) {
    switch(dtype) {
        case Dtype::Int16:    return f(typename dtype_traits<Dtype::Int16>::type{});
        case Dtype::Int32:    return f(typename dtype_traits<Dtype::Int32>::type{});
        case Dtype::Int64:    return f(typename dtype_traits<Dtype::Int64>::type{});
        case Dtype::Float32:  return f(typename dtype_traits<Dtype::Float32>::type{});
        case Dtype::Float64:  return f(typename dtype_traits<Dtype::Float64>::type{});
        case Dtype::Bfloat16: return f(typename dtype_traits<Dtype::Bfloat16>::type{});
        case Dtype::Float16:  return f(typename dtype_traits<Dtype::Float16>::type{});
        default:
            throw std::runtime_error("Unsupported Dtype");
    }
}

} // namespace OwnTensor

#endif