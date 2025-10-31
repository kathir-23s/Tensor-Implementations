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
            
            // Matrix dimensions
            size_t m = a_shape[a_ndim - 2];  // rows of A
            size_t n = a_shape[a_ndim - 1];  // cols of A (inner dim)
            size_t p = b_shape[b_ndim - 1];  // cols of B
            
            // Batch dimensions
            size_t batch_dims = out_ndim - 2;
            
            // Iterate over all batch dimensions
            std::vector<size_t> batch_idx(batch_dims, 0);
            
            while (true) 
            {
                // Calculate offsets for current batch
                size_t a_batch_offset = 0;
                size_t b_batch_offset = 0;
                size_t out_batch_offset = 0;
                
                for (size_t i = 0; i < batch_dims; ++i) 
                {
                    size_t a_dim_idx = (i < a_ndim - 2 && a_shape[i] > 1) ? batch_idx[batch_dims - 1 - i] : 0;
                    size_t b_dim_idx = (i < b_ndim - 2 && b_shape[i] > 1) ? batch_idx[batch_dims - 1 - i] : 0;
                    
                    a_batch_offset += a_dim_idx * a_strides[i];
                    b_batch_offset += b_dim_idx * b_strides[i];
                    out_batch_offset += batch_idx[i] * out_strides[i];
                }
                
                // Perform 2D matmul for current batch
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
                
            // Loops and counts down from (batch_dims - 1) to 0
            // This logic correctly increments the batch counter and signals when to stop.
            bool all_batches_processed = true;
            for (size_t dim = batch_dims; dim-- > 0; ) 
            {
                batch_idx[dim]++;
                if (batch_idx[dim] < static_cast<size_t>(out_shape[dim])) {
                    // This dimension was successfully incremented without wrapping around.
                    // This means there are more batches to process.
                    all_batches_processed = false;
                    break; // Exit the for-loop
                }
                // This dimension overflowed. Reset it to 0 and continue to the next
                // (more significant) dimension to perform a "carry".
                batch_idx[dim] = 0;
            }

            // If the for-loop completed without finding another batch to process,
            // the flag will remain true, and we can exit the main while-loop.
                if (all_batches_processed) 
                {
                    break; // This breaks the 'while(true)'
                }
            }
        });
    }
}
            

// Old while loop logic and it is replaced with a better approach that takes for just 2D matrices without batches better
            // Increment batch indices
        //     size_t dim = batch_dims - 1;
        //     while (dim >= 0) {
        //         batch_idx[dim]++;
        //         if (batch_idx[dim] < out_shape[dim]) {
        //             break;
        //         }
        //         batch_idx[dim] = 0;
        //         dim--;
        //     }
        //     if (dim < 0) break;
        // }});