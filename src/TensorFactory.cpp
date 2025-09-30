#include "../include/Tensor.h"
#include <cstring>
#include <random>

Tensor Tensor::zeros(Shape shape, TensorOptions opts) 
{
    Tensor tensor(shape, opts);
    dispatch_by_dtype(opts.dtype, [&](auto dummy)
    {
        using T = decltype(dummy);
        tensor.fill(T(0));
    });
    return tensor;
}

Tensor Tensor::ones(Shape shape, TensorOptions opts)
{
    Tensor tensor(shape, opts);

    dispatch_by_dtype(opts.dtype, [&](auto dummy)
    {
        using T = decltype(dummy);
        tensor.fill(T(1));
    });

    return tensor;
}

Tensor Tensor::full(Shape shape, TensorOptions opts, float value)
{
    Tensor tensor(shape, opts);

    dispatch_by_dtype(opts.dtype, [&](auto dummy)
    {
        using T = decltype(dummy);
        tensor.fill(T(value));
    });
    return tensor;
}



