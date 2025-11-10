// tensor_kernels/cpu/ConditionalOps.cpp
#include "core/Tensor.h"
#include "dtype/Types.h"
#include <cstddef>

namespace OwnTensor {

namespace {
    // Helper: check if condition is true (nonzero)
    template<typename CondT>
    inline bool is_true(const CondT* ptr, size_t idx) {
        return ptr[idx] != static_cast<CondT>(0);
    }

    // Generic where implementation
    template<typename CondT, typename DataT>
    void apply_where_cpu(const CondT* cond, const DataT* input,
                         const DataT* other, DataT* out, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = is_true(cond, i) ? input[i] : other[i];
        }
    }
}

void cpu_where(const Tensor& condition, const Tensor& input,
               const Tensor& other, Tensor& out) {
    const size_t n = condition.numel();
    const Dtype cond_dtype = condition.dtype();
    const Dtype data_dtype = input.dtype();
    
    // Dispatch on condition type
    if (cond_dtype == Dtype::Int32) {
        const int32_t* cond_ptr = condition.data<int32_t>();
        
        // Dispatch on data type
        if (data_dtype == Dtype::Float32) {
            apply_where_cpu(cond_ptr, input.data<float>(),
                           other.data<float>(), out.data<float>(), n);
        } else if (data_dtype == Dtype::Float64) {
            apply_where_cpu(cond_ptr, input.data<double>(),
                           other.data<double>(), out.data<double>(), n);
        } else if (data_dtype == Dtype::Int32) {
            apply_where_cpu(cond_ptr, input.data<int32_t>(),
                           other.data<int32_t>(), out.data<int32_t>(), n);
        } else if (data_dtype == Dtype::Int64) {
            apply_where_cpu(cond_ptr, input.data<int64_t>(),
                           other.data<int64_t>(), out.data<int64_t>(), n);
        } else {
            throw std::runtime_error("cpu_where: unsupported data dtype");
        }
    } else if (cond_dtype == Dtype::Int64) {
        const int64_t* cond_ptr = condition.data<int64_t>();
        
        if (data_dtype == Dtype::Float32) {
            apply_where_cpu(cond_ptr, input.data<float>(),
                           other.data<float>(), out.data<float>(), n);
        } else if (data_dtype == Dtype::Float64) {
            apply_where_cpu(cond_ptr, input.data<double>(),
                           other.data<double>(), out.data<double>(), n);
        } else if (data_dtype == Dtype::Int32) {
            apply_where_cpu(cond_ptr, input.data<int32_t>(),
                           other.data<int32_t>(), out.data<int32_t>(), n);
        } else if (data_dtype == Dtype::Int64) {
            apply_where_cpu(cond_ptr, input.data<int64_t>(),
                           other.data<int64_t>(), out.data<int64_t>(), n);
        } else {
            throw std::runtime_error("cpu_where: unsupported data dtype");
        }
    } else {
        throw std::runtime_error("cpu_where: condition must be Int32 or Int64");
    }
}

} // namespace OwnTensor
