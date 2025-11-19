#pragma once

#include "core/Tensor.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor
{

void cpu_matmul(const Tensor& A, const Tensor& B, Tensor& output)
{
    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        const T* b_ptr = B.data<T>();
        T* out_ptr = output.data<T>();

        const auto& a_shape = A.shape().dims;
        const auto& b_shape = B.shape().dims;
        const auto& out_shape = output.shape().dims;

        const auto& a_strides = A.stride().strides;
        const auto& b_strides = B.stride().strides;
        const auto& out_strides = output.stride().strides;

        size_t a_ndim = a_shape.size();
        size_t b_ndim = b_shape.size();
        size_t out_ndim = out_shape.size();

        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        size_t batch_dims = out_ndim - 2;
        std::vector<size_t> batch_idx(batch_dims, 0);

        while (true)
        {
            size_t a_batch_offset = 0;
            size_t b_batch_offset = 0;
            size_t out_batch_offset = 0;

            for (size_t i = 0; i < batch_dims; ++i)
            {
                size_t out_idx = batch_idx[i];
                out_batch_offset += out_idx * out_strides[i];

                size_t a_batch_dim = i - (batch_dims - (a_ndim - 2));
                size_t b_batch_dim = i - (batch_dims - (b_ndim - 2));

                if (i >= batch_dims - (a_ndim - 2)) {
                    size_t a_dim = a_shape[a_batch_dim];
                    size_t a_idx = (a_dim > 1) ? out_idx : 0;
                    a_batch_offset += a_idx * a_strides[a_batch_dim];
                }

                if (i >= batch_dims - (b_ndim - 2)) {
                    size_t b_dim = b_shape[b_batch_dim];
                    size_t b_idx = (b_dim > 1) ? out_idx : 0;
                    b_batch_offset += b_idx * b_strides[b_batch_dim];
                }
            }

            for (size_t i = 0; i < m; ++i)
            {
                for (size_t j = 0; j < p; ++j)
                {
                    T sum{};
                    for (size_t k = 0; k < n; ++k)
                    {
                        size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
                        size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
                        sum += a_ptr[a_idx] * b_ptr[b_idx];
                    }
                    size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
                    out_ptr[out_idx] = sum;
                }
            }

            bool all_batches_processed = true;
            for (size_t dim = batch_dims; dim-- > 0;)
            {
                batch_idx[dim]++;
                if (batch_idx[dim] < static_cast<size_t>(out_shape[dim])) {
                    all_batches_processed = false;
                    break;
                }
                batch_idx[dim] = 0;
            }

            if (all_batches_processed) break;
        }
    });
}
}
            
