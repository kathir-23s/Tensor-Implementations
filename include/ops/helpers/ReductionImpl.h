#pragma once

#ifndef OWNTENSOR_REDUCTIONS_IMPL_H
#define OWNTENSOR_REDUCTIONS_IMPL_H

#include "core/Tensor.h" 
#include "dtype/Types.h" 
#include "ops/helpers/ReductionUtils.h" 
#include "ops/helpers/ReductionOps.h" 
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <omp.h>

namespace OwnTensor {
namespace detail {

// =================================================================
// HELPER: Check if we should use double accumulation for better precision
// =================================================================
template <typename T>
constexpr bool should_use_double_accumulation() {
    return std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;
}

// =================================================================
// --- CORE REDUCTION KERNEL (TENSOR -> TENSOR) ---
// =================================================================

template <typename T, template <typename> class OpType, typename AccT = T>
Tensor reduce_kernel(
    const Tensor& input, 
    const std::vector<int64_t>& normalized_axes, 
    const Shape& output_shape) 
{
    using Op = OpType<T>;

    // 1. Determine output dtype
    Dtype output_dtype = input.dtype();
    
    if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
        // Index reductions always output Int64
        output_dtype = Dtype::Int64;
    } else if constexpr (std::is_integral_v<T>) {
        // Integer reductions widen to Int64
        output_dtype = Dtype::Int64;
    } 
    
    Tensor output({output_shape}, TensorOptions().with_dtype(output_dtype).with_req_grad(false));

    // 2. Setup
    const T* input_data = input.data<T>();
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    
    const int64_t reduced_count = calculate_reduced_count(input_dims, normalized_axes);
    
    if (reduced_count == 0 && input.numel() > 0) {
        throw std::runtime_error("Reduction error: reduced count is zero but input is non-empty.");
    }
    
    // Determine output C++ type
    using OutputCppT = typename std::conditional<
        std::is_same_v<AccT, ValueIndex<T>>, 
        int64_t,
        typename std::conditional<
            std::is_integral_v<T>,
            int64_t,
            T
        >::type
    >::type;
    
    OutputCppT* output_data = output.data<OutputCppT>(); 

    Op op;
    const int64_t num_slices = output.numel();
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    // Calculate reduced_dims once
    std::vector<int64_t> reduced_dims;
    for(size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }

    // =================================================================
    // Use double accumulation for FP16/BF16 for maximum precision
    // =================================================================
    using AccumulatorT = typename std::conditional<
        std::is_same_v<AccT, ValueIndex<T>>,
        ValueIndex<T>,  
        typename std::conditional<
            should_use_double_accumulation<T>(),
            double,  // FP16/BF16 use double accumulation
            typename std::conditional<
                std::is_integral_v<T>,
                int64_t,  // Integers use int64_t accumulation
                T         // FP32/FP64 use their own type
            >::type
        >::type
    >::type;

    // =================================================================
    // Kahan summation for floating point sum operations (numerical stability)
    // =================================================================
    constexpr bool use_kahan = std::is_same_v<OpType<T>, SumOp<T>> && 
                               !std::is_same_v<AccT, ValueIndex<T>> &&
                               (std::is_floating_point_v<AccumulatorT> || 
                                std::is_same_v<AccumulatorT, double>);

    // 3. Parallel execution
    #pragma omp parallel for
    for (int64_t output_index = 0; output_index < num_slices; ++output_index) 
    {
        if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
            // =========================================================
            // INDEX REDUCTIONS PATH (argmax, argmin, etc.)
            // =========================================================
            ValueIndex<T> accumulator = op.identity();
            std::vector<int64_t> out_coords = unravel_index(output_index, output_shape.dims);

            for (int64_t i = 0; i < reduced_count; ++i) {
                std::vector<int64_t> slice_coords = detail::unravel_index(i, reduced_dims); 
                std::vector<int64_t> full_input_coords(input_dims.size());
                int out_coord_idx = 0;
                int slice_coord_idx = 0;
                
                for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                    bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                    if (is_reduced) {
                        full_input_coords[dim] = slice_coords[slice_coord_idx++];
                    } else {
                        if (rank_preserved) {
                            full_input_coords[dim] = out_coords[dim];
                        } else {
                            full_input_coords[dim] = out_coords[out_coord_idx];
                        }
                        out_coord_idx++;
                    }
                }
                
                int64_t input_lin_idx = ravel_index(full_input_coords, input_strides);
                T input_value = input_data[input_lin_idx];
                ValueIndex<T> current_val_index = {input_value, i};
                accumulator = op.reduce(accumulator, current_val_index);
            }
            
            output_data[output_index] = accumulator.index;
            
        } else {
            // =========================================================
            // VALUE REDUCTIONS PATH (sum, max, mean, etc.)
            // =========================================================
            
            // Kahan state (only used if use_kahan is true)
            AccumulatorT kahan_sum = 0;
            AccumulatorT kahan_c = 0;
            
            // Initialize accumulator
            AccumulatorT accumulator;
            if constexpr (should_use_double_accumulation<T>()) {
                accumulator = static_cast<double>(op.identity());
            } else if constexpr (std::is_integral_v<T>) {
                accumulator = static_cast<int64_t>(op.identity());
            } else {
                accumulator = op.identity();
            }
            
            std::vector<int64_t> out_coords = unravel_index(output_index, output_shape.dims);

            for (int64_t i = 0; i < reduced_count; ++i) {
                std::vector<int64_t> slice_coords = detail::unravel_index(i, reduced_dims); 
                std::vector<int64_t> full_input_coords(input_dims.size());
                int out_coord_idx = 0;
                int slice_coord_idx = 0;
                
                for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                    bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                    if (is_reduced) {
                        full_input_coords[dim] = slice_coords[slice_coord_idx++];
                    } else {
                        if (rank_preserved) {
                            full_input_coords[dim] = out_coords[dim];
                        } else {
                            full_input_coords[dim] = out_coords[out_coord_idx];
                        }
                        out_coord_idx++;
                    }
                }
                
                int64_t input_lin_idx = ravel_index(full_input_coords, input_strides);
                T input_value = input_data[input_lin_idx];

                if constexpr (use_kahan) {
        // Kahan summation for maximum numerical stability
        AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
        
        // CRITICAL: Overflow/NaN detection for numerical stability
        if (std::isinf(kahan_sum) || std::isnan(kahan_sum)) {
            kahan_sum += val_acc;  // Fallback to simple accumulation
        } else {
            AccumulatorT y = val_acc - kahan_c;
            AccumulatorT t = kahan_sum + y;
            kahan_c = (t - kahan_sum) - y;
            kahan_sum = t;
        }
    }  else {
                    // Standard accumulation
                    AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
                    accumulator = op.reduce(accumulator, val_acc);
                }
            }
            
            // CRITICAL: Safe conversion back to output type
            // Let natural overflow/underflow happen - float_to_float16 handles it correctly
            if constexpr (use_kahan) {
                if constexpr (std::is_same_v<T, float16_t>) {
                    // Convert via float, let float_to_float16 handle overflow→inf
                    output_data[output_index] = static_cast<OutputCppT>(static_cast<T>(static_cast<float>(kahan_sum)));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(static_cast<T>(static_cast<float>(kahan_sum)));
                } else {
                    output_data[output_index] = static_cast<OutputCppT>(kahan_sum);
                }
            } else {
                if constexpr (std::is_same_v<T, float16_t>) {
                    // Convert via float, let float_to_float16 handle overflow→inf
                    output_data[output_index] = static_cast<OutputCppT>(static_cast<T>(static_cast<float>(accumulator)));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(static_cast<T>(static_cast<float>(accumulator)));
                } else {
                    output_data[output_index] = static_cast<OutputCppT>(accumulator);
                }
            }
        }
    }

    return output;
}

// =================================================================
// --- DISPATCHER TEMPLATES ---
// =================================================================

template <typename T, template <typename> class OpType>
Tensor dispatch_reduction(const Tensor& input, const std::vector<int64_t>& normalized_axes, bool keepdim) {
    
    if constexpr (std::is_same_v<OpType<T>, ArgMaxOp<T>> || 
                  std::is_same_v<OpType<T>, ArgMinOp<T>> || 
                  std::is_same_v<OpType<T>, NanArgMaxOp<T>> || 
                  std::is_same_v<OpType<T>, NanArgMinOp<T>>) 
    {
        Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
        return reduce_kernel<T, OpType, ValueIndex<T>>(input, normalized_axes, output_shape);
    } 
    else 
    {
        Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
        return reduce_kernel<T, OpType, T>(input, normalized_axes, output_shape);
    }
}

// =================================================================
// --- MEAN REDUCTION DISPATCHER ---
// =================================================================

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_kernel(const Tensor& input, const std::vector<int64_t>& normalized_axes, bool keepdim) {
    
    int64_t reduced_count = detail::calculate_reduced_count(input.shape().dims, normalized_axes);

    if (reduced_count == 0) {
        throw std::runtime_error("Cannot compute mean: reduced count is zero.");
    }

    Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

    if constexpr (std::is_integral_v<T>) {
        // Integers output Float64
        Tensor output({output_shape}, TensorOptions().with_dtype(Dtype::Float64).with_req_grad(false));
        
        const T* input_data = input.data<T>();
        const std::vector<int64_t>& input_dims = input.shape().dims;
        const std::vector<int64_t>& input_strides = input.stride().strides;
        
        const int64_t num_slices = output.numel();
        const bool rank_preserved = input_dims.size() == output_shape.dims.size();
        
        std::vector<int64_t> reduced_dims;
        for(size_t dim = 0; dim < input_dims.size(); ++dim) {
            bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
            if (is_reduced) {
                reduced_dims.push_back(input_dims[dim]);
            }
        }
        
        double* output_data = output.data<double>();
        SumOpType<T> op;
        
        #pragma omp parallel for
        for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
            double accumulator = 0.0;
            
            std::vector<int64_t> out_coords = unravel_index(output_index, output_shape.dims);
            
            for (int64_t i = 0; i < reduced_count; ++i) {
                std::vector<int64_t> slice_coords = detail::unravel_index(i, reduced_dims);
                std::vector<int64_t> full_input_coords(input_dims.size());
                int out_coord_idx = 0;
                int slice_coord_idx = 0;
                
                for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                    bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                    if (is_reduced) {
                        full_input_coords[dim] = slice_coords[slice_coord_idx++];
                    } else {
                        if (rank_preserved) {
                            full_input_coords[dim] = out_coords[dim];
                        } else {
                            full_input_coords[dim] = out_coords[out_coord_idx];
                        }
                        out_coord_idx++;
                    }
                }
                
                int64_t input_lin_idx = ravel_index(full_input_coords, input_strides);
                T input_value = input_data[input_lin_idx];
                
                accumulator += static_cast<double>(input_value);
            }
            
            output_data[output_index] = accumulator / static_cast<double>(reduced_count);
        }
        
        return output;
        
    } else {
        // Floating point: use double accumulation for FP16/BF16
        using AccT = typename std::conditional<
            should_use_double_accumulation<T>(),
            double,  
            T        
        >::type;
        
        Tensor sum_result = reduce_kernel<T, SumOpType, AccT>(input, normalized_axes, output_shape);
        T* sum_data = sum_result.data<T>();
        
        // Divide by count to get mean
        if constexpr (should_use_double_accumulation<T>()) {
            const double divisor_d = static_cast<double>(reduced_count);
            
            #pragma omp parallel for
            for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
                double val_d = static_cast<double>(sum_data[i]);
                val_d /= divisor_d;
                
                // Safe conversion with clamping for FP16
                if constexpr (std::is_same_v<T, float16_t>) {
                    double clamped = std::max(-65504.0, std::min(65504.0, val_d));
                    sum_data[i] = static_cast<T>(static_cast<float>(clamped));
                } else {
                    sum_data[i] = static_cast<T>(static_cast<float>(val_d));
                }
            }
        } else {
            const T divisor = static_cast<T>(reduced_count);
            
            #pragma omp parallel for
            for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
                sum_data[i] /= divisor;
            }
        }

        return sum_result;
    }
}

} // namespace detail
} // namespace OwnTensor
#endif // OWNTENSOR_REDUCTIONS_IMPL_H