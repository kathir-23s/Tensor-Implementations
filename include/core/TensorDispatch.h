#pragma once

#ifndef TENSOR_DISPATCH_H
#define TENSOR_DISPATCH_H

#include "core/Tensor.h"
#include "dtype/DtypeTraits.h" // Needed if using type_to_dtype or is_float/is_integral
#include <stdexcept>
#include <type_traits> // Required for std::decay_t

namespace OwnTensor { // <<< START OF THE OWNTENSOR NAMESPACE


template<typename Func>
static auto dispatch_by_dtype(Dtype dtype, Func&& f) {
    // Determine the C++ type associated with the Dtype and execute the lambda 'f'
    switch(dtype) {
        // Integer types
        case Dtype::Int16:    return f(typename dtype_traits<Dtype::Int16>::type{});
        case Dtype::Int32:    return f(typename dtype_traits<Dtype::Int32>::type{});
        case Dtype::Int64:    return f(typename dtype_traits<Dtype::Int64>::type{});
            
        // Floating point types
        case Dtype::Float32:  return f(typename dtype_traits<Dtype::Float32>::type{});
        case Dtype::Float64:  return f(typename dtype_traits<Dtype::Float64>::type{});
            
        // 16-bit float types (Note: Actual computation might require promotion 
        // to float/double, but here we pass the underlying C++ type for the kernel).
        // For our current implementation, the underlying C++ type is uint16_t (see dtype/DtypeTraits.h)
        case Dtype::Bfloat16: return f(typename dtype_traits<Dtype::Bfloat16>::type{}); 
        case Dtype::Float16:  return f(typename dtype_traits<Dtype::Float16>::type{}); 
            
        default:
            throw std::runtime_error("Unsupported Dtype encountered in dispatch_by_dtype.");
    }
}

} // <<< END OF OwnTensor NAMESPACE

#endif // TENSOR_DISPATCH_H
