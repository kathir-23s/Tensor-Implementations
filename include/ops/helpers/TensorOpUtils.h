#include "core/Tensor.h"
#include "ops/helpers/BroadcastUtils.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor 
{
    template <typename func>
    void apply_binary_operation(const Tensor& A, const Tensor& B, Tensor& output, func op)
    {
        // if (A.dtype() != B.dtype() || A.dtype() != output.dtype())
        // {
        //     throw std::runtime_error("Tensor datatypes are not matching");
        // }
        
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
            }  
            else 
            {
                // N-dimensional broadcasting
                const auto& a_shape = A.shape().dims;
                const auto& b_shape = B.shape().dims;
                const auto& out_shape = output.shape().dims;
                
                const auto& a_strides = A.stride().strides;
                const auto& b_strides = B.stride().strides;
                [[maybe_unused]] const auto& out_strides = output.stride().strides;
                
                size_t out_ndim = out_shape.size();
                
                // Calculate broadcast strides
                // Calculate broadcast strides - CORRECTED
            std::vector<size_t> a_bcast_strides(out_ndim, 0);
            std::vector<size_t> b_bcast_strides(out_ndim, 0);

            size_t a_ndim = a_shape.size();
            size_t b_ndim = b_shape.size();

            for (size_t i = 0; i < out_ndim; ++i) {
                size_t a_dim_idx = a_ndim - out_ndim + i;
                size_t b_dim_idx = b_ndim - out_ndim + i;
                
                if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
                    a_bcast_strides[i] = a_strides[a_dim_idx];
                }
                if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
                    b_bcast_strides[i] = b_strides[b_dim_idx];
                }
            }
                
                // Iterate over output elements
                std::vector<size_t> coords(out_ndim, 0);
                
                for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
                    // Calculate coordinates - CORRECTED
                    size_t temp_idx = linear_idx;
                    for (int dim = out_ndim - 1; dim >= 0; --dim) {
                        coords[dim] = temp_idx % out_shape[dim];
                        temp_idx /= out_shape[dim];
                    }
                    
                    // Calculate input indices using broadcast strides
                    size_t a_idx = 0;
                    size_t b_idx = 0;
                    
                    for (size_t dim = 0; dim < out_ndim; ++dim) {
                        a_idx += coords[dim] * a_bcast_strides[dim];
                        b_idx += coords[dim] * b_bcast_strides[dim];
                    }
                    
                    output_ptr[linear_idx] = op(a_ptr[a_idx], b_ptr[b_idx]);
                }
            }
        });
    }

    
    template <typename func>
    void apply_binary_op_bool(const Tensor& A, const Tensor& B, Tensor& output, func op)
    {
        // if (A.dtype() != B.dtype() )
        // {
        //     throw std::runtime_error("Tensor datatypes are not matching");
        // }
        
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
                const auto& a_shape = A.shape().dims;
                const auto& b_shape = B.shape().dims;
                const auto& out_shape = output.shape().dims;
                
                const auto& a_strides = A.stride().strides;
                const auto& b_strides = B.stride().strides;
                [[maybe_unused]] const auto& out_strides = output.stride().strides;
                
                size_t out_ndim = out_shape.size();
                
                // Calculate broadcast strides
                // Calculate broadcast strides - CORRECTED
            std::vector<size_t> a_bcast_strides(out_ndim, 0);
            std::vector<size_t> b_bcast_strides(out_ndim, 0);

            size_t a_ndim = a_shape.size();
            size_t b_ndim = b_shape.size();

            for (size_t i = 0; i < out_ndim; ++i) {
                size_t a_dim_idx = a_ndim - out_ndim + i;
                size_t b_dim_idx = b_ndim - out_ndim + i;
                
                if (a_dim_idx < a_ndim && a_shape[a_dim_idx] > 1) {
                    a_bcast_strides[i] = a_strides[a_dim_idx];
                }
                if (b_dim_idx < b_ndim && b_shape[b_dim_idx] > 1) {
                    b_bcast_strides[i] = b_strides[b_dim_idx];
                }
            }
                
                // Iterate over output elements
                std::vector<size_t> coords(out_ndim, 0);
                
                for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
                    // Calculate coordinates - CORRECTED
                    size_t temp_idx = linear_idx;
                    for (int dim = out_ndim - 1; dim >= 0; --dim) {
                        coords[dim] = temp_idx % out_shape[dim];
                        temp_idx /= out_shape[dim];
                    }
                    
                    // Calculate input indices using broadcast strides
                    size_t a_idx = 0;
                    size_t b_idx = 0;
                    
                    for (size_t dim = 0; dim < out_ndim; ++dim) {
                        a_idx += coords[dim] * a_bcast_strides[dim];
                        b_idx += coords[dim] * b_bcast_strides[dim];
                    }
                    
                    output_ptr[linear_idx] = op(a_ptr[a_idx], b_ptr[b_idx]);
                }
            }
        });
}

template <typename func>
    void apply_not_bool(const Tensor& A, Tensor& output, func op)
    {
        size_t total_elems = output.numel();
        
        dispatch_by_dtype(A.dtype(), [&](auto dummy)
        {
            using T = decltype(dummy);
            const T* a_ptr = A.data<T>();
            bool* output_ptr = output.data<bool>();

          // Same shape - direct element-wise operation
                for (size_t i = 0; i < total_elems; ++i) {
                    output_ptr[i] = op(a_ptr[i]);
                }
            
           
        });
}

}
