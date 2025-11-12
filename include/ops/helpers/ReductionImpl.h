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

// #ifdef WITH_CUDA
// #include "ReductionImplGPU.h" 
// #endif


namespace OwnTensor {
namespace detail {
// Forward declarations only (no implementation here!)
template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input, 
                               const std::vector<int64_t>& normalized_axes, 
                               bool keepdim, cudaStream_t stream);//✨✨✨

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input, 
                                     const std::vector<int64_t>& normalized_axes, 
                                     bool keepdim, cudaStream_t stream);//✨✨✨

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor& input, 
                         const std::vector<int64_t>& normalized_axes, 
                         bool keepdim, cudaStream_t stream);//✨✨✨
 template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_gpu(const Tensor& input, 
                             const std::vector<int64_t>& normalized_axes, 
                             bool keepdim,
                             int64_t correction,
                             cudaStream_t stream);//✨✨✨                        
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
    if constexpr (std::is_same_v<T, bool>) {
        output_dtype = Dtype::Bool;  // ✅ Boolean operations return Bool
    } else if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
        // Index reductions always output Int64
        output_dtype = Dtype::Int64;
    } else if constexpr (std::is_integral_v<T>) {
        // Integer reductions widen to Int64
        output_dtype = Dtype::Int64;
    } 
    
    Tensor output({output_shape}, TensorOptions().with_dtype(output_dtype).with_device(input.device()).with_req_grad(input.requires_grad()));

    // 2. Setup
    const T* input_data = input.data<T>();
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    
    const int64_t reduced_count = calculate_reduced_count(input_dims, normalized_axes);
    
    if (reduced_count == 0 && input.numel() > 0) {
        throw std::runtime_error("Reduction error: reduced count is zero but input has " + 
                                std::to_string(input.numel()) + " elements.");
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
 constexpr size_t MAX_DIMS = 16;
    const size_t ndim = output_shape.dims.size();
    
    if (ndim > MAX_DIMS) {
        throw std::runtime_error("Tensor rank exceeds maximum supported dimensions (16)");
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
          // ✅ STACK-ALLOCATED BUFFER (no heap allocation)
        int64_t out_coords_buf[MAX_DIMS];
        unravel_index_stack(output_index, output_shape.dims.data(), ndim, out_coords_buf);
        if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
            // =========================================================
            // INDEX REDUCTIONS PATH (argmax, argmin, etc.)
            // =========================================================
            ValueIndex<T> accumulator = op.identity();
          
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
                            full_input_coords[dim] = out_coords_buf[dim];
                        } else {
                            full_input_coords[dim] = out_coords_buf[out_coord_idx];
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
            
           // std::vector<int64_t> out_coords = unravel_index(output_index, output_shape.dims);(commented out as no use)
           // out_coords_buf already computed above

            if constexpr (use_kahan) {
                // Kahan state and initialization (used only for SumOp)
                AccumulatorT kahan_sum = 0;
                AccumulatorT kahan_c = 0;
                
                // Kahan Loop
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
                                full_input_coords[dim] = out_coords_buf[dim];
                            } else {
                                full_input_coords[dim] = out_coords_buf[out_coord_idx];
                            }
                            out_coord_idx++;
                        }
                    }
                    
                    int64_t input_lin_idx = ravel_index(full_input_coords, input_strides);
                    T input_value = input_data[input_lin_idx];

                    // Kahan summation for maximum numerical stability
                    AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
                    
                    // Overflow/NaN detection for numerical stability
                    if (std::isinf(kahan_sum) || std::isnan(kahan_sum)) {
                        kahan_sum += val_acc;  // Fallback to simple accumulation
                    } else {
                        AccumulatorT y = val_acc - kahan_c;
                        AccumulatorT t = kahan_sum + y;
                        kahan_c = (t - kahan_sum) - y;
                        kahan_sum = t;
                    }
                }
                
                // =================================================================
                // CRITICAL: Safe conversion back to output type (Kahan path)
                // =================================================================
                if constexpr (std::is_same_v<T, float16_t>) {
                    // FP16: overflow→inf handled by float_to_float16
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(kahan_sum))
                    );
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    // BF16: overflow→inf handled by float_to_bfloat16
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(kahan_sum))
                    );
                } else {
                    output_data[output_index] = static_cast<OutputCppT>(kahan_sum);
                }

            } else {
                // Initialize standard accumulator (used for all other reductions)
                AccumulatorT accumulator;
                    // ✅ FIX: Don't cast identity for index reductions (ValueIndex type)
                if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
                    // This path should never be reached since index reductions
                    // are handled in the first if branch above
                    accumulator = op.identity();
                } else if constexpr (should_use_double_accumulation<T>()) {
                    accumulator = static_cast<double>(op.identity());
                } else if constexpr (std::is_integral_v<T>) {
                    accumulator = static_cast<int64_t>(op.identity());
                } else {
                    accumulator = op.identity();
                }

                // Standard Loop
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
                                full_input_coords[dim] = out_coords_buf[dim];
                            } else {
                                full_input_coords[dim] = out_coords_buf[out_coord_idx];
                            }
                            out_coord_idx++;
                        }
                    }
                    
                    int64_t input_lin_idx = ravel_index(full_input_coords, input_strides);
                    T input_value = input_data[input_lin_idx];

                      // ✅ FIX: Proper type conversion based on AccumulatorT
                    if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
                        // Should never reach here - handled above
                    } else {
                        AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
                        accumulator = op.reduce(accumulator, val_acc);
                    }
                }
                

                
                // =================================================================
                // CRITICAL: Safe conversion back to output type (Standard path)
                // =================================================================
                if constexpr (std::is_same_v<T, float16_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(accumulator))
                    );
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(accumulator))
                    );
                } else {
                    output_data[output_index] = static_cast<OutputCppT>(accumulator);
                }
            }
        }
    }

    return output;
}


// =================================================================
// --- DISPATCHER TEMPLATES WITH TYPE VALIDATION ---
// =================================================================

template <typename T, template <typename> class OpType>                                                 
Tensor dispatch_reduction(const Tensor& input, const std::vector<int64_t>& normalized_axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    constexpr bool is_all_any_op = 
        std::is_same_v<OpType<T>, AllOp<T>> ||
        std::is_same_v<OpType<T>, AnyOp<T>>;
    
    if constexpr (is_all_any_op && !std::is_same_v<T, bool>) {
        // Convert non-Bool tensor to Bool tensor (0 → false, non-zero → true)
        Tensor bool_input = input.to_bool();  // You need to implement this
        
        // Now call the Bool version
        return dispatch_reduction<bool, OpType>(bool_input, normalized_axes, keepdim, stream);
    }
    // ✅ CRITICAL: Validate that NaN operations are only used with floating point types
    constexpr bool is_nan_op = 
        std::is_same_v<OpType<T>, NanSumOp<T>> ||
        std::is_same_v<OpType<T>, NanProductOp<T>> ||
        std::is_same_v<OpType<T>, NanMinOp<T>> ||
        std::is_same_v<OpType<T>, NanMaxOp<T>> ||
        std::is_same_v<OpType<T>, NanArgMinOp<T>> ||
        std::is_same_v<OpType<T>, NanArgMaxOp<T>>;
    
    constexpr bool is_float_type = 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> ||
        std::is_same_v<T, float16_t> ||
        std::is_same_v<T, bfloat16_t>;
    
    // Block NaN operations on non-float types at compile time
    if constexpr (is_nan_op && !is_float_type) {
        throw std::runtime_error(
            "NaN-aware operations are only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
             "Got: " + get_dtype_name(input.dtype())
        );
    }
    
#ifdef WITH_CUDA
    if (input.is_cuda()) {
        // Route to GPU implementation
   
        if constexpr (std::is_same_v<OpType<T>, ArgMaxOp<T>> || 
                      std::is_same_v<OpType<T>, ArgMinOp<T>> || 
                      std::is_same_v<OpType<T>, NanArgMaxOp<T>> || 
                      std::is_same_v<OpType<T>, NanArgMinOp<T>>) 
        {
            return dispatch_index_reduction_gpu<T, OpType>(input, normalized_axes, keepdim, stream);//✨✨✨
        } 
        else 
        {
            return dispatch_reduction_gpu<T, OpType>(input, normalized_axes, keepdim, stream);//✨✨✨
        }
    }
#endif

    // CPU path continues as before
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
// --- MEAN REDUCTION DISPATCHER WITH TYPE VALIDATION ---
// =================================================================

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_kernel(const Tensor& input, const std::vector<int64_t>& normalized_axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    
    // ✅ CRITICAL: Validate NaN-aware mean operations
    constexpr bool is_nan_sum = std::is_same_v<SumOpType<T>, NanSumOp<T>>;
    constexpr bool is_float_type = 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> ||
        std::is_same_v<T, float16_t> ||
        std::is_same_v<T, bfloat16_t>;
    
    if constexpr (is_nan_sum && !is_float_type) { 
        throw std::runtime_error(
            "NaN-aware mean is only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
             "Got: " + get_dtype_name(input.dtype())
        );
    }
    
#ifdef WITH_CUDA
    if (input.is_cuda()) {
        return dispatch_mean_gpu<T, SumOpType>(input, normalized_axes, keepdim, stream);//✨✨✨
    }
#endif

    // CPU implementation continues as before...
    int64_t reduced_count = detail::calculate_reduced_count(input.shape().dims, normalized_axes);

    if (reduced_count == 0) {
        throw std::runtime_error("Cannot compute mean: reduced count is zero.");
    }

    Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

    if constexpr (std::is_integral_v<T>) {
        // Integers output Float64
        Tensor output({output_shape}, TensorOptions().with_dtype(Dtype::Float64).with_device(input.device()).with_req_grad(input.requires_grad()));
        
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
        //SumOpType<T> op;
        
        #pragma omp parallel for
        for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
            int64_t accumulator = 0;
            
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
                
                accumulator += input_value;
            }
            
            output_data[output_index] = static_cast<double>(accumulator) / static_cast<double>(reduced_count);
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
    
    using SumT = typename std::conditional<
        should_use_double_accumulation<T>(),
        double,  
        T        
    >::type;
    
    T* sum_data = sum_result.data<T>();
    
    // ✅ FIX: For NaN-aware mean, count only non-NaN values
    SumT divisor;
    if constexpr (is_nan_sum) {
        // Count non-NaN values in the input tensor
        const T* input_data = input.data<T>();
        const std::vector<int64_t>& input_dims = input.shape().dims;
        const std::vector<int64_t>& input_strides = input.stride().strides;
        
        std::vector<int64_t> reduced_dims;
        for(size_t dim = 0; dim < input_dims.size(); ++dim) {
            bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
            if (is_reduced) {
                reduced_dims.push_back(input_dims[dim]);
            }
        }
        
        const bool rank_preserved = input_dims.size() == output_shape.dims.size();
        const int64_t num_slices = sum_result.numel();
        
        // Create a tensor to store valid counts for each output position
        std::vector<int64_t> valid_counts(num_slices, 0);
        
        #pragma omp parallel for
        for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
            int64_t valid_count = 0;
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
                
                // Check if value is not NaN
                if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    if (!std::isnan(static_cast<float>(input_value))) {
                        valid_count++;
                    }
                } else {
                    if (!std::isnan(input_value)) {
                        valid_count++;
                    }
                }
            }
            
            valid_counts[output_index] = valid_count;
        }
        
        // Now divide each sum by its corresponding valid count
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
            if (valid_counts[i] > 0) {
                SumT val = static_cast<SumT>(sum_data[i]);
                val /= static_cast<SumT>(valid_counts[i]);  // Divide by non-NaN count
                
                if constexpr (std::is_same_v<T, float16_t>) {
                    sum_data[i] = static_cast<T>(static_cast<float>(val));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    sum_data[i] = static_cast<T>(static_cast<float>(val));
                } else {
                    sum_data[i] = val;
                }
            } else {
                // All values were NaN - result is NaN
                if constexpr (std::is_same_v<T, float16_t>) {
                    sum_data[i] = static_cast<T>(std::nanf(""));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    sum_data[i] = static_cast<T>(std::nanf(""));
                } else {
                    sum_data[i] = std::numeric_limits<T>::quiet_NaN();
                }
            }
        }
        
    } else {
        // Regular mean: divide by total reduced count
        divisor = static_cast<SumT>(reduced_count);
        
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
            SumT val = static_cast<SumT>(sum_data[i]);
            val /= divisor;

            if constexpr (std::is_same_v<T, float16_t>) {
                sum_data[i] = static_cast<T>(static_cast<float>(val));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                sum_data[i] = static_cast<T>(static_cast<float>(val));
            } else {
                sum_data[i] = val;
            }
        }
    }
    
    return sum_result;


        // Final result must be cast back to the original Tensor type (T) if AccT was double.
        // The reduce_kernel returns a Tensor<T> or Tensor<double>, but the output Dtype is T.
        // The previous code had a bug here.
        // We ensure the output Tensor's data type matches the original T
        // if constexpr (should_use_double_accumulation<T>()) {
        //     Tensor final_output({output_shape}, TensorOptions().with_dtype(input.dtype()).with_req_grad(false));
        //     T* final_output_data = final_output.data<T>();
            
        //     #pragma omp parallel for
        //     for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
        //         // Safe conversion from SumT (double) back to output type (T)
        //         if constexpr (std::is_same_v<T, float16_t>) {
        //             final_output_data[i] = static_cast<T>(static_cast<float>(sum_data[i]));
        //         } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        //             final_output_data[i] = static_cast<T>(static_cast<float>(sum_data[i]));
        //         } else {
        //             final_output_data[i] = static_cast<T>(sum_data[i]);
        //         }
        //     }
        //     return final_output;
        // } else {
        //     return sum_result;
        // }
    }
}

//----------------------------------------------------------------------
// VARIANCE REDUCTION DISPATCHER (Two-pass algorithm)
//----------------------------------------------------------------------

template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_kernel(const Tensor& input, 
                                const std::vector<int64_t>& normalized_axes, 
                                bool keepdim,
                                int64_t correction, cudaStream_t stream) {
    // Determine if this is NaN-aware variance
    if constexpr (std::is_same_v<T, bool>) {
        throw std::runtime_error(
             "reduce_var: Bool dtype not supported for statistical operations."
        );
    }
    constexpr bool is_nan_aware = std::is_same_v<VarianceOpType<T>, NanVarianceOp<T>>;
    
    constexpr bool is_float_type = 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> ||
        std::is_same_v<T, float16_t> ||
        std::is_same_v<T, bfloat16_t>;
    
    if constexpr (is_nan_aware && !is_float_type) {
        throw std::runtime_error(
            "NaN-aware variance is only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
             "Got: " + get_dtype_name(input.dtype())
        );
    }
#ifdef WITH_CUDA
    if (input.is_cuda()) {
        return dispatch_variance_gpu<T, VarianceOpType>(
            input, normalized_axes, keepdim, correction, stream);
    }
#endif
    
    // ✅ STEP 1: Compute mean with keepdim=true (required for broadcasting)
    Tensor mean_tensor = is_nan_aware 
        ? dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, true, stream)
        : dispatch_mean_kernel<T, SumOp>(input, normalized_axes, true, stream);
    
    // ✅ STEP 2: Calculate output shape and metadata
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);
    
    // Determine output dtype
    Dtype output_dtype;
    if constexpr (std::is_integral_v<T>) {
        output_dtype = Dtype::Float64;
    } else {
        output_dtype = input.dtype();
    }
    
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype)
        .with_device(input.device())
        .with_req_grad(input.requires_grad()));
    
    // ✅ STEP 3: Prepare data pointers
    const T* input_data = input.data<T>();
    
    // ✅ CRITICAL FIX: For integers, mean is stored as double
    using MeanCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    const MeanCppT* mean_data = mean_tensor.data<MeanCppT>();
    
    using AccT = typename std::conditional<
        should_use_double_accumulation<T>(),
        double,
        typename std::conditional<
            std::is_integral_v<T>,
            double,
            T
        >::type
    >::type;
    
    using OutputT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    OutputT* output_data = output.data<OutputT>();
    
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    const int64_t num_slices = output.numel();
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    // Get mean tensor strides for indexing
    const std::vector<int64_t>& mean_strides = mean_tensor.stride().strides;
    
    // Calculate reduced_dims
    std::vector<int64_t> reduced_dims;
    for(size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                         != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    
    // ✅ STEP 4: Compute sum of squared deviations in parallel
    #pragma omp parallel for
    for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
        AccT accumulator = 0;
        int64_t valid_count = 0;  // Only used for NaN-aware variance
        
        // Calculate output coordinates
        std::vector<int64_t> out_coords = unravel_index(output_index, output_shape.dims);
        
        // ✅ Map output coordinates to mean tensor coordinates
        // Since mean was computed with keepdim=true, it has same rank as input
        std::vector<int64_t> mean_coords(input_dims.size());
        int out_coord_idx = 0;
        
        for (size_t dim = 0; dim < input_dims.size(); ++dim) {
            bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                             != normalized_axes.end();
            if (is_reduced) {
                mean_coords[dim] = 0;  // Mean tensor has size 1 in reduced dimensions
            } else {
                if (rank_preserved) {
                    mean_coords[dim] = out_coords[dim];
                } else {
                    mean_coords[dim] = out_coords[out_coord_idx];
                }
                out_coord_idx++;
            }
        }
        
        // Get the pre-computed mean value for this slice
        int64_t mean_lin_idx = ravel_index(mean_coords, mean_strides);
        AccT mean_val = static_cast<AccT>(mean_data[mean_lin_idx]);
        
        // Check if mean is NaN
        bool mean_is_nan = false;
        if constexpr (std::is_floating_point_v<T> || 
                      std::is_same_v<T, float16_t> || 
                      std::is_same_v<T, bfloat16_t>) {
            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                mean_is_nan = std::isnan(static_cast<float>(mean_val));
            } else {
                mean_is_nan = std::isnan(mean_val);
            }
        }
        
        // ✅ Accumulate squared deviations
        for (int64_t i = 0; i < reduced_count; ++i) {
            std::vector<int64_t> slice_coords = detail::unravel_index(i, reduced_dims);
            std::vector<int64_t> full_input_coords(input_dims.size());
            out_coord_idx = 0;
            int slice_coord_idx = 0;
            
            for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                                 != normalized_axes.end();
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
            
            // Check if value is NaN
            bool is_nan = false;
            if constexpr (std::is_floating_point_v<T> || 
                          std::is_same_v<T, float16_t> || 
                          std::is_same_v<T, bfloat16_t>) {
                if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    is_nan = std::isnan(static_cast<float>(input_value));
                } else {
                    is_nan = std::isnan(input_value);
                }
            }
            
            if constexpr (is_nan_aware) {
                // ✅ NaN-aware: Skip NaN values and count valid ones
                if (!is_nan) {
                    AccT val_acc = static_cast<AccT>(input_value);
                    AccT diff = val_acc - mean_val;
                    accumulator += diff * diff;
                    valid_count++;  // Only increment for valid values
                }
            } else {
                // ✅ Regular variance: Propagate NaN immediately
                if (is_nan || mean_is_nan) {
                    accumulator = std::numeric_limits<AccT>::quiet_NaN();
                    break;  // Early exit on NaN
                }
                
                AccT val_acc = static_cast<AccT>(input_value);
                AccT diff = val_acc - mean_val;
                accumulator += diff * diff;
                // Don't increment valid_count here!
            }
        }
        
        // ✅ STEP 5: Compute divisor and variance
        int64_t divisor;
        if constexpr (is_nan_aware) {
            divisor = valid_count - correction;  // Use counted valid values
        } else {
            divisor = reduced_count - correction;  // Use total count
        }
        
        // Compute final variance
        AccT variance;
        if (std::isnan(accumulator)) {
            variance = accumulator;
        } else if (divisor <= 0) {
            variance = std::numeric_limits<AccT>::quiet_NaN();
        } else {
            variance = accumulator / static_cast<AccT>(divisor);
        }
        
        // ✅ STEP 6: Convert back to output type
        if constexpr (std::is_same_v<T, float16_t>) {
            output_data[output_index] = static_cast<OutputT>(
                static_cast<T>(static_cast<float>(variance))
            );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            output_data[output_index] = static_cast<OutputT>(
                static_cast<T>(static_cast<float>(variance))
            );
        } else {
            output_data[output_index] = static_cast<OutputT>(variance);
        }
    }
    
    return output;
}
} // namespace detail
} // namespace OwnTensor
#endif // OWNTENSOR_REDUCTIONS_IMPL_H


/* OPTIMIZED SINGLE-PASS NaN-AWARE MEAN (KEPT FOR REFERENCE)
} else {
    // Floating point: use double accumulation for FP16/BF16
    using AccT = typename std::conditional<
        should_use_double_accumulation<T>(),
        double,  
        T        
    >::type;
    
    // ✅ OPTIMIZATION: For NaN-aware mean, compute sum AND count in ONE pass
    if constexpr (is_nan_sum) {
        // Build sum result manually with NaN counting
        Tensor output({output_shape}, TensorOptions().with_dtype(input.dtype()).with_req_grad(false));
        T* output_data = output.data<T>();
        
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
        
        // ✅ SINGLE PASS: Accumulate sum AND count non-NaN values
        #pragma omp parallel for
        for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
            AccT accumulator = 0;
            int64_t valid_count = 0;
            
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
                
                // Check if not NaN and accumulate
                bool is_valid;
                if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    is_valid = !std::isnan(static_cast<float>(input_value));
                } else {
                    is_valid = !std::isnan(input_value);
                }
                
                if (is_valid) {
                    accumulator += static_cast<AccT>(input_value);
                    valid_count++;
                }
            }
            
            // Compute mean
            if (valid_count > 0) {
                AccT mean_val = accumulator / static_cast<AccT>(valid_count);
                
                // Convert back to output type
                if constexpr (std::is_same_v<T, float16_t>) {
                    output_data[output_index] = static_cast<T>(static_cast<float>(mean_val));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    output_data[output_index] = static_cast<T>(static_cast<float>(mean_val));
                } else {
                    output_data[output_index] = static_cast<T>(mean_val);
                }
            } else {
                // All values were NaN
                output_data[output_index] = static_cast<T>(std::nanf(""));
            }
        }
        
        return output;
        
    } else {
        // Regular mean: use reduce_kernel for sum, then divide
        Tensor sum_result = reduce_kernel<T, SumOpType, AccT>(input, normalized_axes, output_shape);
        
        using SumT = typename std::conditional<
            should_use_double_accumulation<T>(),
            double,  
            T        
        >::type;
        
        T* sum_data = sum_result.data<T>();
        const SumT divisor = static_cast<SumT>(reduced_count);
        
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
            SumT val = static_cast<SumT>(sum_data[i]);
            val /= divisor;

            if constexpr (std::is_same_v<T, float16_t>) {
                sum_data[i] = static_cast<T>(static_cast<float>(val));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                sum_data[i] = static_cast<T>(static_cast<float>(val));
            } else {
                sum_data[i] = val;
            }
        }
        
        return sum_result;
    }
}

*/