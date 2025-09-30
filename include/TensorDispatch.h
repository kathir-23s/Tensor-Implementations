#pragma once

#ifndef TENSOR_DISPATCH_H
#define TENSOR_DISPATCH_H

#include "Tensor.h"

template<typename Func>
static void dispatch_by_dtype(Dtype dtype, Func&& f) {
    switch(dtype) {
        case Dtype::Int16: f(int16_t{}); break;
        case Dtype::Int32: f(int32_t{}); break;
        case Dtype::Int64: f(int64_t{}); break;
        case Dtype::Float32: f(float{}); break;
        case Dtype::Float64: f(double{}); break;
        default: throw std::runtime_error("Unsupported dtype");
    }
}


#endif