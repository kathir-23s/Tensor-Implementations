// src/UnaryOps/ReductionKernels.cu - COMPLETE GPU INTRINSIC OPTIMIZED VERSION
#include "ReductionKernels.cuh"
#include "datatypes/Dtype.h"  
#include "datatypes/DtypeTraits.h"

#include "ReductionOps.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <limits> // Needed for std::numeric_limits<double>::quiet_NaN()

namespace OwnTensor {
namespace cuda {

/* ===========================================================
// ✅ OPTIMIZED: NATIVE GPU TYPE CONVERSION HELPERS
// ===========================================================

template<typename T>
__device__ __host__ inline float load_and_convert(const T* ptr, int64_t idx) {
    if constexpr (std::is_same_v<T, float16_t>) {
        return __half2float(ptr[idx]);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return __bfloat162float(ptr[idx]);
    } else {
        return static_cast<float>(ptr[idx]);
    }
}

template<typename T>
__device__ __host__ inline void convert_and_store(T* ptr, int64_t idx, float val) {
    if constexpr (std::is_same_v<T, float16_t>) {
        ptr[idx] = __float2half(val);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        ptr[idx] = __float2bfloat16(val);
    } else {
        ptr[idx] = static_cast<T>(val);
    }
}
*/
// ===========================================================
// ✅ OPTIMIZED: WARP-LEVEL REDUCTIONS WITH INTRINSICS
// ===========================================================

// Generic warp reduction (for integers, or unspecialized types)
template<typename T, template<typename> class OpType>
__device__ T warp_reduce(T val, OpType<T> op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = op.reduce(val, other);
    }
    return val;
}

// ✅ NEW: Specialized warp reduction for FLOAT (using intrinsics)
template<template<typename> class OpType>
__device__ float warp_reduce(float val, OpType<float> op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        
        // ✅ Use intrinsics based on operation type
        if constexpr (std::is_same_v<OpType<float>, detail::SumOp<float>>) {
            val = __fadd_rn(val, other);
        } else if constexpr (std::is_same_v<OpType<float>, detail::ProductOp<float>>) {
            val = __fmul_rn(val, other);
        } else if constexpr (std::is_same_v<OpType<float>, detail::MaxOp<float>>) {
            val = fmaxf(val, other);
        } else if constexpr (std::is_same_v<OpType<float>, detail::MinOp<float>>) {
            val = fminf(val, other);
        } else {
            // Fallback for NaN-aware ops or other custom ops
            val = op.reduce(val, other);
        }
    }
    return val;
}

// ✅ NEW: Specialized warp reduction for DOUBLE
template<template<typename> class OpType>
__device__ double warp_reduce(double val, OpType<double> op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(0xffffffff, val, offset);
        
        if constexpr (std::is_same_v<OpType<double>, detail::SumOp<double>>) {
            val = __dadd_rn(val, other);
        } else if constexpr (std::is_same_v<OpType<double>, detail::ProductOp<double>>) {
            val = __dmul_rn(val, other);
        } else if constexpr (std::is_same_v<OpType<double>, detail::MaxOp<double>>) {
            val = fmax(val, other);
        } else if constexpr (std::is_same_v<OpType<double>, detail::MinOp<double>>) {
            val = fmin(val, other);
        } else {
            val = op.reduce(val, other);
        }
    }
    return val;
}

// ✅ NEW: Specialized warp reduction for INT64
template<template<typename> class OpType>
__device__ int64_t warp_reduce(int64_t val, OpType<int64_t> op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int64_t other = __shfl_down_sync(0xffffffff, val, offset);
        
        // Integer ops don't have intrinsics, use manual
        if constexpr (std::is_same_v<OpType<int64_t>, detail::SumOp<int64_t>>) {
            val += other;
        } else if constexpr (std::is_same_v<OpType<int64_t>, detail::ProductOp<int64_t>>) {
            val *= other;
        } else if constexpr (std::is_same_v<OpType<int64_t>, detail::MaxOp<int64_t>>) {
            val = (val > other) ? val : other;
        } else if constexpr (std::is_same_v<OpType<int64_t>, detail::MinOp<int64_t>>) {
            val = (val < other) ? val : other;
        } else {
            val = op.reduce(val, other);
        }
    }
    return val;
}

// ===========================================================
// ✅ OPTIMIZED: BLOCK-LEVEL REDUCTIONS (TEMPLATED FOR ALL TYPES)
// ===========================================================

// ✅ FIXED: Generic template for all types
template<typename AccT, typename T, template<typename> class OpType>
__device__ AccT block_reduce(AccT val, AccT* shared, OpType<T> op) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 1. Warp reduction - use appropriate type
    if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
        // For reduced precision types with float accumulator
        OpType<float> float_op;
        val = warp_reduce(val, float_op);
    } else {
        // For other types
        OpType<AccT> acc_op;
        val = warp_reduce(val, acc_op);
    }

    // 2. Write warp results to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 3. Last warp reduces results from shared memory
    if (wid == 0) {
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            OpType<float> float_op;
            val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : float_op.identity();
            val = warp_reduce(val, float_op);
        } else {
            OpType<AccT> acc_op;
            val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : acc_op.identity();
            val = warp_reduce(val, acc_op);
        }
    }

    return val;
}
// ===========================================================
// ✅ OPTIMIZED: MAIN VALUE REDUCTION KERNEL
// ===========================================================

template<typename T, typename OutputT, template<typename> class OpType>
__global__ void reduce_kernel(
    const T* __restrict__ input_data,
    OutputT* __restrict__ output_data,
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved)
{
    OpType<T> op;
    
    constexpr bool is_reduced_precision_v = 
        std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;
    
    constexpr bool is_integer_type = std::is_integral_v<T>;
    constexpr bool is_integer_sum = std::is_integral_v<T> && 
        std::is_same_v<OpType<T>, detail::SumOp<T>>;
    constexpr bool is_integer_product = std::is_integral_v<T> && 
        std::is_same_v<OpType<T>, detail::ProductOp<T>>;

    // ✅ ACCUMULATOR TYPE SELECTION
    using AccumulatorType = typename std::conditional_t<
        is_integer_sum || is_integer_product,
        int64_t,
        typename std::conditional_t<
            is_reduced_precision_v,
            float,
            T
        >
    >;

    extern __shared__ char shared_mem[];
    AccumulatorType* shared = reinterpret_cast<AccumulatorType*>(shared_mem);

    // ===========================================================
    // ✅ OPTIMIZED ACCUMULATION LOOP WITH INTRINSICS
    // ===========================================================
    
    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        AccumulatorType accumulator;
        
        if constexpr (is_integer_sum || is_integer_product) {
            accumulator = (is_integer_sum) ? 0LL : 1LL;
        } else if constexpr (is_reduced_precision_v) {
            // For reduced precision, the float accumulator starts at the identity for float
            // Note: This relies on op.identity() for reduced precision being a float identity
            // but for Sum/Product it's hardcoded for safety/simplicity
            if constexpr (std::is_same_v<OpType<T>, detail::SumOp<T>>) {
                accumulator = 0.0f;
            } else if constexpr (std::is_same_v<OpType<T>, detail::ProductOp<T>>) {
                accumulator = 1.0f;
            } else {
                accumulator = op.identity(); // Should be a float
            }
        } else {
            accumulator = op.identity();
        }

        int64_t out_coords[10];
        int64_t temp = output_index;
        for (int d = num_reduced_dims - 1; d >= 0; --d) {
            out_coords[d] = temp % output_dims[d];
            temp /= output_dims[d];
        }

        // ✅ OPTIMIZED: Use intrinsics in accumulation loop
        for (int64_t i = threadIdx.x; i < reduced_count; i += blockDim.x) {
            int64_t slice_coords[10];
            int64_t tmp = i;
            for (int d = num_reduced_dims - 1; d >= 0; --d) {
                slice_coords[d] = tmp % reduced_dims[d];
                tmp /= reduced_dims[d];
            }

            int64_t full_input_coords[10];
            int out_coord_idx = 0;
            int slice_coord_idx = 0;

            for (int dim = 0; dim < ndim; ++dim) {
                bool is_reduced = false;
                for (int ax = 0; ax < num_axes; ++ax) {
                    if (normalized_axes[ax] == dim) {
                        is_reduced = true;
                        break;
                    }
                }

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

            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }

            T input_value = input_data[input_lin_idx];

            // ✅ CRITICAL FIX: Use intrinsics for accumulation
            if constexpr (is_reduced_precision_v) {
                // FP16/BF16 path: convert to float and use __fadd_rn, etc.
                float val_f = load_and_convert(&input_data[0], input_lin_idx);
                
                if constexpr (std::is_same_v<OpType<T>, detail::SumOp<T>>) {
                    accumulator = __fadd_rn(accumulator, val_f);
                } else if constexpr (std::is_same_v<OpType<T>, detail::ProductOp<T>>) {
                    accumulator = __fmul_rn(accumulator, val_f);
                } else if constexpr (std::is_same_v<OpType<T>, detail::MaxOp<T>>) {
                    accumulator = fmaxf(accumulator, val_f);
                } else if constexpr (std::is_same_v<OpType<T>, detail::MinOp<T>>) {
                    accumulator = fminf(accumulator, val_f);
                } else {
                    // NaN-aware ops use op.reduce (which typically performs checks)
                    accumulator = op.reduce(accumulator, val_f);
                }
                
            } else if constexpr (is_integer_sum) {
                accumulator += static_cast<int64_t>(input_value);
                
            } else if constexpr (is_integer_product) {
                accumulator *= static_cast<int64_t>(input_value);
                
            } else if constexpr (std::is_same_v<T, float>) {
                // FP32 path: use intrinsics
                if constexpr (std::is_same_v<OpType<T>, detail::SumOp<T>>) {
                    accumulator = __fadd_rn(accumulator, input_value);
                } else if constexpr (std::is_same_v<OpType<T>, detail::ProductOp<T>>) {
                    accumulator = __fmul_rn(accumulator, input_value);
                } else if constexpr (std::is_same_v<OpType<T>, detail::MaxOp<T>>) {
                    accumulator = fmaxf(accumulator, input_value);
                } else if constexpr (std::is_same_v<OpType<T>, detail::MinOp<T>>) {
                    accumulator = fminf(accumulator, input_value);
                } else {
                    accumulator = op.reduce(accumulator, input_value);
                }
                
            } else if constexpr (std::is_same_v<T, double>) {
                // FP64 path: use intrinsics
                if constexpr (std::is_same_v<OpType<T>, detail::SumOp<T>>) {
                    accumulator = __dadd_rn(accumulator, input_value);
                } else if constexpr (std::is_same_v<OpType<T>, detail::ProductOp<T>>) {
                    accumulator = __dmul_rn(accumulator, input_value);
                } else if constexpr (std::is_same_v<OpType<T>, detail::MaxOp<T>>) {
                    accumulator = fmax(accumulator, input_value);
                } else if constexpr (std::is_same_v<OpType<T>, detail::MinOp<T>>) {
                    accumulator = fmin(accumulator, input_value);
                } else {
                    accumulator = op.reduce(accumulator, input_value);
                }
                
            } else {
                // Generic path (for other integer max/min, or unspecialized types)
                accumulator = op.reduce(accumulator, input_value);
            }
        }

        // ===========================================================
        // ✅ BLOCK REDUCTION (WITH INTRINSICS IN WARP REDUCE)
        // ===========================================================

        if constexpr (is_integer_sum || is_integer_product) {
            // INTEGER PATH: Custom reduction for int64_t for sum/product
            int lane = threadIdx.x % 32;
            int wid = threadIdx.x / 32;

            // Warp-level reduction (manual implementation to avoid calling templated warp_reduce)
            for (int offset = 16; offset > 0; offset /= 2) {
                int64_t other = __shfl_down_sync(0xffffffff, accumulator, offset);
                if constexpr (is_integer_sum) {
                    accumulator += other;
                } else {
                    accumulator *= other;
                }
            }

            if (lane == 0) shared[wid] = accumulator;
            __syncthreads();

            // Block-level reduction
            if (wid == 0) {
                accumulator = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 
                             (is_integer_sum ? 0LL : 1LL);
                
                for (int offset = 16; offset > 0; offset /= 2) {
                    int64_t other = __shfl_down_sync(0xffffffff, accumulator, offset);
                    if constexpr (is_integer_sum) {
                        accumulator += other;
                    } else {
                        accumulator *= other;
                    }
                }
            }

            if (threadIdx.x == 0) {
                output_data[output_index] = static_cast<OutputT>(accumulator);
            }

        } else {
            // FLOAT PATH: Use optimized block reduction (calls warp_reduce with intrinsics)
            AccumulatorType final_val = block_reduce<AccumulatorType, T, OpType>(accumulator, shared, op);

            if (threadIdx.x == 0) {
                if constexpr (std::is_same_v<AccumulatorType, OutputT>) {
                    output_data[output_index] = final_val;
                } else if constexpr (is_reduced_precision_v) {
                    convert_and_store(output_data, output_index, final_val);
                } else {
                    output_data[output_index] = static_cast<OutputT>(final_val);
                }
            }
        }
    }
}

// ===========================================================
// INDEX REDUCTION KERNEL (ARGMAX/ARGMIN) - UNCHANGED
// ===========================================================

template<typename T, template<typename> class OpType>
__global__ void reduce_index_kernel(
    const T* __restrict__ input_data,
    int64_t* __restrict__ output_data,
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved)
{
    OpType<T> op;
    using ValueIndexType = detail::ValueIndex<T>;

    extern __shared__ char shared_mem[];
    ValueIndexType* shared = reinterpret_cast<ValueIndexType*>(shared_mem);

    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        ValueIndexType accumulator = op.identity();

        int64_t out_coords[10];
        int64_t temp = output_index;
        for (int d = num_reduced_dims - 1; d >= 0; --d) {
            out_coords[d] = temp % output_dims[d];
            temp /= output_dims[d];
        }

        for (int64_t i = threadIdx.x; i < reduced_count; i += blockDim.x) {
            int64_t slice_coords[10];
            int64_t tmp = i;
            for (int d = num_reduced_dims - 1; d >= 0; --d) {
                slice_coords[d] = tmp % reduced_dims[d];
                tmp /= reduced_dims[d];
            }

            int64_t full_input_coords[10];
            int out_coord_idx = 0;
            int slice_coord_idx = 0;

            for (int dim = 0; dim < ndim; ++dim) {
                bool is_reduced = false;
                for (int ax = 0; ax < num_axes; ++ax) {
                    if (normalized_axes[ax] == dim) {
                        is_reduced = true;
                        break;
                    }
                }

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

            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }

            T input_value = input_data[input_lin_idx];
            ValueIndexType current_val_index = {input_value, i};
            accumulator = op.reduce(accumulator, current_val_index);
        }

        // Block reduction (simple warp reduction without specialized intrinsics for ValueIndex)
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        for (int offset = 16; offset > 0; offset /= 2) {
            ValueIndexType other;
            // Note: ValueIndex requires separate shfl for value and index
            other.value = __shfl_down_sync(0xffffffff, accumulator.value, offset);
            other.index = __shfl_down_sync(0xffffffff, accumulator.index, offset);
            accumulator = op.reduce(accumulator, other);
        }

        if (lane == 0) shared[wid] = accumulator;
        __syncthreads();

        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared[lane] : op.identity();

            for (int offset = 16; offset > 0; offset /= 2) {
                ValueIndexType other;
                other.value = __shfl_down_sync(0xffffffff, accumulator.value, offset);
                other.index = __shfl_down_sync(0xffffffff, accumulator.index, offset);
                accumulator = op.reduce(accumulator, other);
            }
        }

        if (threadIdx.x == 0) {
            output_data[output_index] = accumulator.index;
        }
    }
}

// ===========================================================
// ✅ OPTIMIZED: MEAN REDUCTION KERNEL
// ===========================================================

template<typename T, template<typename> class SumOpType>
__global__ void reduce_mean_kernel(
    const T* __restrict__ input_data,
    T* __restrict__ output_data,
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved)
{
    constexpr bool is_nan_aware = 
        std::is_same_v<SumOpType<T>, detail::NanSumOp<T>>;
    
    constexpr bool is_reduced_precision = 
        std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

    // Shared memory for block reduction: double for sum, int64_t for count (if nan-aware)
    extern __shared__ char shared_mem[];
    double* shared_acc = reinterpret_cast<double*>(shared_mem);
    int64_t* shared_count = reinterpret_cast<int64_t*>(shared_acc + blockDim.x / 32);

    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        // Accumulate in double for better precision
        double accumulator = 0.0;
        int64_t valid_count = 0; // Only used for NaN-aware mean

        int64_t out_coords[10];
        int64_t temp = output_index;
        for (int d = num_reduced_dims - 1; d >= 0; --d) {
            out_coords[d] = temp % output_dims[d];
            temp /= output_dims[d];
        }

        // ✅ OPTIMIZED: Accumulation with intrinsics
        for (int64_t i = threadIdx.x; i < reduced_count; i += blockDim.x) {
            int64_t slice_coords[10];
            int64_t tmp = i;
            for (int d = num_reduced_dims - 1; d >= 0; --d) {
                slice_coords[d] = tmp % reduced_dims[d];
                tmp /= reduced_dims[d];
            }

            int64_t full_input_coords[10];
            int out_coord_idx = 0;
            int slice_coord_idx = 0;

            for (int dim = 0; dim < ndim; ++dim) {
                bool is_reduced = false;
                for (int ax = 0; ax < num_axes; ++ax) {
                    if (normalized_axes[ax] == dim) {
                        is_reduced = true;
                        break;
                    }
                }

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

            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }

            T input_value = input_data[input_lin_idx];

            double val_d;
            if constexpr (is_reduced_precision) {
                val_d = static_cast<double>(load_and_convert(&input_data[0], input_lin_idx));
            } else {
                val_d = static_cast<double>(input_value);
            }


            if constexpr (is_nan_aware) {
                if (!isnan(val_d)) {
                    accumulator = __dadd_rn(accumulator, val_d);
                    valid_count++;
                }
            } else {
                // ✅ Use intrinsics for accumulation (double add)
                accumulator = __dadd_rn(accumulator, val_d);
            }
        }

        // Warp-level reduction using intrinsics
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        for (int offset = 16; offset > 0; offset /= 2) {
            double other_acc = __shfl_down_sync(0xffffffff, accumulator, offset);
            accumulator = __dadd_rn(accumulator, other_acc);
            
            if constexpr (is_nan_aware) {
                int64_t other_count = __shfl_down_sync(0xffffffff, valid_count, offset);
                valid_count += other_count;
            }
        }

        if (lane == 0) {
            shared_acc[wid] = accumulator;
            if constexpr (is_nan_aware) {
                shared_count[wid] = valid_count;
            }
        }
        __syncthreads();

        // Block-level reduction
        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared_acc[lane] : 0.0;
            if constexpr (is_nan_aware) {
                valid_count = (threadIdx.x < blockDim.x / 32) ? shared_count[lane] : 0;
            }

            for (int offset = 16; offset > 0; offset /= 2) {
                double other_acc = __shfl_down_sync(0xffffffff, accumulator, offset);
                accumulator = __dadd_rn(accumulator, other_acc);
                
                if constexpr (is_nan_aware) {
                    int64_t other_count = __shfl_down_sync(0xffffffff, valid_count, offset);
                    valid_count += other_count;
                }
            }
        }

        if (threadIdx.x == 0) {
            double mean_val;
            
            if constexpr (is_nan_aware) {
                if (valid_count == 0) {
                    mean_val = std::numeric_limits<double>::quiet_NaN();
                } else {
                    mean_val = accumulator / static_cast<double>(valid_count);
                }
            } else {
                mean_val = accumulator / static_cast<double>(reduced_count);
            }

            if constexpr (is_reduced_precision) {
                convert_and_store(output_data, output_index, static_cast<float>(mean_val));
            } else {
                output_data[output_index] = static_cast<T>(mean_val);
            }
        }
    }
}

// ===========================================================
// EXPLICIT TEMPLATE INSTANTIATIONS (ALL 7 DTYPES)
// ===========================================================

#define INSTANTIATE_REDUCE_KERNEL(T, OutputT, OpType) \
    template __global__ void reduce_kernel<T, OutputT, OpType>( \
        const T*, OutputT*, const int64_t*, const int64_t*, const int64_t*, \
        const int64_t*, const int64_t*, int64_t, int64_t, int, int, int, bool);

#define INSTANTIATE_INDEX_KERNEL(T, OpType) \
    template __global__ void reduce_index_kernel<T, OpType>( \
        const T*, int64_t*, const int64_t*, const int64_t*, const int64_t*, \
        const int64_t*, const int64_t*, int64_t, int64_t, int, int, int, bool);

#define INSTANTIATE_MEAN_KERNEL(T, SumOpType) \
    template __global__ void reduce_mean_kernel<T, SumOpType>( \
        const T*, T*, const int64_t*, const int64_t*, const int64_t*, \
        const int64_t*, const int64_t*, int64_t, int64_t, int, int, int, bool);

// Integer types (accumulate in int64_t)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::MaxOp)

INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::MaxOp)

INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::MaxOp)

// Float16 types (FP16/BF16 accumulate in float)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanMaxOp)

INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanMaxOp)

// Float32 types
INSTANTIATE_REDUCE_KERNEL(float, float, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanMaxOp)

// Float64 types
INSTANTIATE_REDUCE_KERNEL(double, double, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanMaxOp)

// Index reductions (all types)
INSTANTIATE_INDEX_KERNEL(int16_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(int16_t, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(int32_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(int32_t, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(int64_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(int64_t, detail::ArgMaxOp)

INSTANTIATE_INDEX_KERNEL(float16_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::NanArgMaxOp)

INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::NanArgMaxOp)

INSTANTIATE_INDEX_KERNEL(float, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(float, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(float, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(float, detail::NanArgMaxOp)

INSTANTIATE_INDEX_KERNEL(double, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(double, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(double, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(double, detail::NanArgMaxOp)

// Mean operations (all types)
INSTANTIATE_MEAN_KERNEL(int16_t, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(int32_t, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(int64_t, detail::SumOp)

INSTANTIATE_MEAN_KERNEL(float16_t, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(float16_t, detail::NanSumOp)

INSTANTIATE_MEAN_KERNEL(bfloat16_t, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(bfloat16_t, detail::NanSumOp)

INSTANTIATE_MEAN_KERNEL(float, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(float, detail::NanSumOp)

INSTANTIATE_MEAN_KERNEL(double, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(double, detail::NanSumOp)

} // namespace cuda
} // namespace OwnTensor