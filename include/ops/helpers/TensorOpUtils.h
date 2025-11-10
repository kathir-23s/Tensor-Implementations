#include "core/Tensor.h"
#include <stdexcept>

namespace OwnTensor 
{
    template <typename func>
    void apply_binary_operation(const Tensor& A, const Tensor& B, Tensor& output, func op)
    {
        if (A.dtype() != B.dtype() || A.dtype() != output.dtype())
        {
            throw std::runtime_error("Tensor datatypes are not matching");
        }
        
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = output.numel();
        
        dispatch_by_dtype(A.dtype(), [&](auto dummy)
        {
            using T = decltype(dummy);
            const T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            T* output_ptr = output.data<T>();

            if (!needs_broadcasting) 
            {
                // Same shape - direct element-wise operation
                for (size_t i = 0; i < total_elems; ++i) {
                    output_ptr[i] = op(a_ptr[i], b_ptr[i]);
                }
            }  else 
            {

            if (A.shape().dims.size() != 2 || B.shape().dims.size() != 2) {
                throw std::runtime_error("Only 2D tensor broadcasting supported");
            }

            // 2D broadcasting with single loop
            size_t a_rows = A.shape().dims[0];
            size_t a_cols = A.shape().dims[1];
            size_t b_rows = B.shape().dims[0];
            size_t b_cols = B.shape().dims[1];
            [[maybe_unused]] size_t out_rows = output.shape().dims[0];
            size_t out_cols = output.shape().dims[1];
            
            // ADD VALIDATION HERE
            bool rows_compatible = (a_rows == b_rows) || (a_rows == 1) || (b_rows == 1);
            bool cols_compatible = (a_cols == b_cols) || (a_cols == 1) || (b_cols == 1);
            
            if (!rows_compatible || !cols_compatible) 
            {
                throw std::runtime_error("Shapes are not broadcastable");
            }

            // Calculate strides for broadcasting
            size_t a_row_stride = (a_rows == 1) ? 0 : a_cols;
            size_t a_col_stride = (a_cols == 1) ? 0 : 1;
            size_t b_row_stride = (b_rows == 1) ? 0 : b_cols;
            size_t b_col_stride = (b_cols == 1) ? 0 : 1;
            
            for (size_t idx = 0; idx < total_elems; ++idx) {
                // Convert linear index to 2D coordinates
                size_t i = idx / out_cols;
                size_t j = idx % out_cols;
                
                // Calculate source indices using strides
                size_t a_idx = (i * a_row_stride) + (j * a_col_stride);
                size_t b_idx = (i * b_row_stride) + (j * b_col_stride);
                
                output_ptr[idx] = op(a_ptr[a_idx], b_ptr[b_idx]);
            }
        }
    });
}

    template <typename func>
    void apply_binary_op_bool(const Tensor& A, const Tensor& B, Tensor& output, func op)
    {
        if (A.dtype() != B.dtype() )
        {
            throw std::runtime_error("Tensor datatypes are not matching");
        }
        
        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        size_t total_elems = output.numel();
        
        dispatch_by_dtype(A.dtype(), [&](auto dummy)
        {
            using T = decltype(dummy);
            const T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            bool* output_ptr = output.data<bool>();

            if (!needs_broadcasting) 
            {
                // Same shape - direct element-wise operation
                for (size_t i = 0; i < total_elems; ++i) {
                    output_ptr[i] = op(a_ptr[i], b_ptr[i]);
                }
            }  
            else 
            {
                throw std::runtime_error("Broadcasting not implemented in apply_binary_op_bool");
            }
        });
    }
}