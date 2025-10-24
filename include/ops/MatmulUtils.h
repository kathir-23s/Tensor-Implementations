#pragma once

#include "core/Tensor.h"
#include <stdexcept>

namespace OwnTensor
{

    void cpu_matmul(const Tensor& A, const Tensor& B, Tensor& output) {
        dispatch_by_dtype(A.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            const T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            T* out_ptr = output.data<T>();
            
            size_t a_rows = A.shape().dims[0];
            size_t a_cols = A.shape().dims[1];
            size_t b_cols = B.shape().dims[1];
            
            size_t a_row_stride = A.stride().strides[0];
            size_t a_col_stride = A.stride().strides[1];
            size_t b_row_stride = B.stride().strides[0];
            size_t b_col_stride = B.stride().strides[1];
            size_t out_row_stride = output.stride().strides[0];
            size_t out_col_stride = output.stride().strides[1];
            
            for (size_t i = 0; i < a_rows; ++i) {
                for (size_t j = 0; j < b_cols; ++j) {
                    T sum{};
                    for (size_t k = 0; k < a_cols; ++k) {
                        size_t a_idx = i * a_row_stride + k * a_col_stride;
                        size_t b_idx = k * b_row_stride + j * b_col_stride;
                        sum += a_ptr[a_idx] * b_ptr[b_idx];
                    }
                    size_t out_idx = i * out_row_stride + j * out_col_stride;
                    out_ptr[out_idx] = sum;
                }
            }
        });
    }
}