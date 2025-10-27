// include/ops/helpers/ReductionKernels.cuh - FIXED SHUFFLE OPERATIONS
#pragma once

#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ReductionOps.h"
#include <limits>

// ✅ Use custom structs everywhere (no native CUDA types)
#include "dtype/Types.h"

namespace OwnTensor {
namespace cuda {

// ═══════════════════════════════════════════════════════════
// WARP SHUFFLE WRAPPERS (AVOID NAMESPACE AMBIGUITY)
// ═══════════════════════════════════════════════════════════

// Generic wrapper - uses global namespace to avoid ADL issues
template<typename T>
__device__ inline T shfl_down(T val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

// Explicit overloads for standard types
__device__ inline int16_t shfl_down(int16_t val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

__device__ inline int32_t shfl_down(int32_t val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

__device__ inline int64_t shfl_down(int64_t val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

__device__ inline float shfl_down(float val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

__device__ inline double shfl_down(double val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

// Custom types - convert through float
__device__ inline OwnTensor::bfloat16_t shfl_down(OwnTensor::bfloat16_t val, unsigned int delta) {
    float f = static_cast<float>(val);
    f = ::__shfl_down_sync(0xffffffff, f, delta, 32);
    return OwnTensor::bfloat16_t(f);
}

__device__ inline OwnTensor::float16_t shfl_down(OwnTensor::float16_t val, unsigned int delta) {
    float f = static_cast<float>(val);
    f = ::__shfl_down_sync(0xffffffff, f, delta, 32);
    return OwnTensor::float16_t(f);
}

// ═══════════════════════════════════════════════════════════
// HELPER TYPE TRAITS
// ═══════════════════════════════════════════════════════════

template <typename T>
constexpr bool is_reduced_precision_v = 
    std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>;

template <typename T>
constexpr bool is_any_float_v = 
    std::is_floating_point_v<T> || is_reduced_precision_v<T>;

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION HELPERS (NO INTRINSICS)
// ═══════════════════════════════════════════════════════════

template<typename T>
__device__ __host__ inline float load_and_convert(const T* ptr, int64_t idx) {
    return static_cast<float>(ptr[idx]);
}

template<typename T>
__device__ __host__ inline void convert_and_store(T* ptr, int64_t idx, float val) {
    ptr[idx] = static_cast<T>(val);
}

// =================================================================
// WARP-LEVEL REDUCTIONS (USING FIXED SHUFFLE)
// =================================================================

template<typename T, template<typename> class OpType>
__device__ T warp_reduce(T val, OpType<T> op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = shfl_down(val, offset);  // ← FIXED
        val = op.reduce(val, other);
    }
    return val;
}

// ===========================================================
// BLOCK-LEVEL REDUCTIONS
// ===========================================================

template<typename AccT, typename T, template<typename> class OpType>
__device__ AccT block_reduce(AccT val, AccT* shared, OpType<T> op) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 1. Warp reduction
    if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
        OpType<float> float_op;
        val = warp_reduce(val, float_op);
    } else {
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

// =================================================================
// MAIN VALUE REDUCTION KERNEL
// =================================================================

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
    
    constexpr bool is_integer_sum = std::is_integral_v<T> && 
        std::is_same_v<OpType<T>, detail::SumOp<T>>;
    constexpr bool is_integer_product = std::is_integral_v<T> && 
        std::is_same_v<OpType<T>, detail::ProductOp<T>>;

    // Accumulator type selection
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

    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        AccumulatorType accumulator;
        
        if constexpr (is_integer_sum || is_integer_product) {
            accumulator = (is_integer_sum) ? 0LL : 1LL;
        } else if constexpr (is_reduced_precision_v) {
            if constexpr (std::is_same_v<OpType<T>, detail::SumOp<T>>) {
                accumulator = 0.0f;
            } else if constexpr (std::is_same_v<OpType<T>, detail::ProductOp<T>>) {
                accumulator = 1.0f;
            } else {
                accumulator = op.identity();
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

        // Accumulation loop (NO INTRINSICS)
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

            // Simple accumulation (no intrinsics)
            if constexpr (is_reduced_precision_v) {
                float val_f = static_cast<float>(input_value);
                accumulator = op.reduce(accumulator, val_f);
            } else if constexpr (is_integer_sum) {
                accumulator += static_cast<int64_t>(input_value);
            } else if constexpr (is_integer_product) {
                accumulator *= static_cast<int64_t>(input_value);
            } else {
                accumulator = op.reduce(accumulator, input_value);
            }
        }

        // Block reduction
        if constexpr (is_integer_sum || is_integer_product) {
            // Integer path
            int lane = threadIdx.x % 32;
            int wid = threadIdx.x / 32;

            for (int offset = 16; offset > 0; offset /= 2) {
                int64_t other = shfl_down(accumulator, offset);  // ← FIXED
                if constexpr (is_integer_sum) {
                    accumulator += other;
                } else {
                    accumulator *= other;
                }
            }

            if (lane == 0) shared[wid] = accumulator;
            __syncthreads();

            if (wid == 0) {
                accumulator = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 
                             (is_integer_sum ? 0LL : 1LL);
                
                for (int offset = 16; offset > 0; offset /= 2) {
                    int64_t other = shfl_down(accumulator, offset);  // ← FIXED
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
            // Float path
            AccumulatorType final_val = block_reduce<AccumulatorType, T, OpType>(accumulator, shared, op);

            if (threadIdx.x == 0) {
                if constexpr (std::is_same_v<AccumulatorType, OutputT>) {
                    output_data[output_index] = final_val;
                } else if constexpr (is_reduced_precision_v) {
                    output_data[output_index] = static_cast<OutputT>(final_val);
                } else {
                    output_data[output_index] = static_cast<OutputT>(final_val);
                }
            }
        }
    }
}

// ===========================================================
// INDEX REDUCTION KERNEL (ARGMAX/ARGMIN) - FIXED
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

        // Block reduction - FIXED SHUFFLE CALLS
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        for (int offset = 16; offset > 0; offset /= 2) {
            ValueIndexType other;
            other.value = shfl_down(accumulator.value, offset);  // ← FIXED
            other.index = shfl_down(accumulator.index, offset);  // ← FIXED
            accumulator = op.reduce(accumulator, other);
        }

        if (lane == 0) shared[wid] = accumulator;
        __syncthreads();

        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared[lane] : op.identity();

            for (int offset = 16; offset > 0; offset /= 2) {
                ValueIndexType other;
                other.value = shfl_down(accumulator.value, offset);  // ← FIXED
                other.index = shfl_down(accumulator.index, offset);  // ← FIXED
                accumulator = op.reduce(accumulator, other);
            }
        }

        if (threadIdx.x == 0) {
            output_data[output_index] = accumulator.index;
        }
    }
}

// ===========================================================
// MEAN REDUCTION KERNEL - FIXED
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

    extern __shared__ char shared_mem[];
    double* shared_acc = reinterpret_cast<double*>(shared_mem);
    int64_t* shared_count = reinterpret_cast<int64_t*>(shared_acc + blockDim.x / 32);

    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        double accumulator = 0.0;
        int64_t valid_count = 0;

        int64_t out_coords[10];
        int64_t temp = output_index;
        for (int d = num_reduced_dims - 1; d >= 0; --d) {
            out_coords[d] = temp % output_dims[d];
            temp /= output_dims[d];
        }

        // Accumulation
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
                val_d = static_cast<double>(static_cast<float>(input_value));
            } else {
                val_d = static_cast<double>(input_value);
            }

            if constexpr (is_nan_aware) {
                if (!isnan(val_d)) {
                    accumulator += val_d;
                    valid_count++;
                }
            } else {
                accumulator += val_d;
            }
        }

        // Warp-level reduction - FIXED SHUFFLE CALLS
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        for (int offset = 16; offset > 0; offset /= 2) {
            double other_acc = shfl_down(accumulator, offset);  // ← FIXED
            accumulator += other_acc;
            
            if constexpr (is_nan_aware) {
                int64_t other_count = shfl_down(valid_count, offset);  // ← FIXED
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

        // Block-level reduction - FIXED SHUFFLE CALLS
        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared_acc[lane] : 0.0;
            if constexpr (is_nan_aware) {
                valid_count = (threadIdx.x < blockDim.x / 32) ? shared_count[lane] : 0;
            }

            for (int offset = 16; offset > 0; offset /= 2) {
                double other_acc = shfl_down(accumulator, offset);  // ← FIXED
                accumulator += other_acc;
                
                if constexpr (is_nan_aware) {
                    int64_t other_count = shfl_down(valid_count, offset);  // ← FIXED
                    valid_count += other_count;
                }
            }
        }

        if (threadIdx.x == 0) {
            double mean_val;
            
            if constexpr (is_nan_aware) {
                if (valid_count == 0) {
                    mean_val = __longlong_as_double(0x7ff8000000000000ULL);
                } else {
                    mean_val = accumulator / static_cast<double>(valid_count);
                }
            } else {
                mean_val = accumulator / static_cast<double>(reduced_count);
            }

            if constexpr (is_reduced_precision) {
                output_data[output_index] = static_cast<T>(static_cast<float>(mean_val));
            } else {
                output_data[output_index] = static_cast<T>(mean_val);
            }
        }
    }
}

} // namespace cuda
} // namespace OwnTensor

#endif // REDUCTION_KERNELS_CUH