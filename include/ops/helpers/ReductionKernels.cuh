// include/ops/helpers/ReductionKernels.cuh - FIXED: Uses NATIVE CUDA types
#pragma once

#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <limits>

// ✅ CRITICAL: Import operation templates (they work on ANY type)
#include "ReductionOps.h"

namespace OwnTensor {
namespace cuda {

// ═══════════════════════════════════════════════════════════
// GPU INTRINSIC HELPERS (NATIVE CUDA TYPES ONLY)
// ═══════════════════════════════════════════════════════════

// ---- Type conversion (use GPU intrinsics) ----
template<typename T> __device__ float to_float(T val) { return static_cast<float>(val); }
template<> __device__ float to_float(__half val) { return __half2float(val); }
template<> __device__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

// ---- From float conversion ----
template<typename T> __device__ T from_float(float val) { return static_cast<T>(val); }
template<> __device__ __half from_float(float val) { return __float2half(val); }
template<> __device__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// // ---- NaN check (use GPU intrinsics) ----
// template<typename T> __device__ bool is_nan(T val) { return isnan(val); }
// template<> __device__ bool is_nan(__half val) { return __hisnan(val); }
// template<> __device__ bool is_nan(__nv_bfloat16 val) { return __hisnan(val); }

// // ---- Arithmetic operations using GPU intrinsics ----
// template<typename T> __device__ T gpu_add(T a, T b) { return a + b; }
// template<> __device__ __half gpu_add(__half a, __half b) { return __hadd(a, b); }
// template<> __device__ __nv_bfloat16 gpu_add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }

// template<typename T> __device__ T gpu_mul(T a, T b) { return a * b; }
// template<> __device__ __half gpu_mul(__half a, __half b) { return __hmul(a, b); }
// template<> __device__ __nv_bfloat16 gpu_mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }

// template<typename T> __device__ bool gpu_lt(T a, T b) { return a < b; }
// template<> __device__ bool gpu_lt(__half a, __half b) { return __hlt(a, b); }
// template<> __device__ bool gpu_lt(__nv_bfloat16 a, __nv_bfloat16 b) { return __hlt(a, b); }

// template<typename T> __device__ bool gpu_gt(T a, T b) { return a > b; }
// template<> __device__ bool gpu_gt(__half a, __half b) { return __hgt(a, b); }
// template<> __device__ bool gpu_gt(__nv_bfloat16 a, __nv_bfloat16 b) { return __hgt(a, b); }

// ═══════════════════════════════════════════════════════════
// WARP SHUFFLE (NATIVE CUDA TYPES)
// ═══════════════════════════════════════════════════════════

template<typename T>
__device__ inline T shfl_down(T val, unsigned int delta) {
    return ::__shfl_down_sync(0xffffffff, val, delta, 32);
}

// ✅ Specialization for __half (uses intrinsic shuffle)
__device__ inline __half shfl_down(__half val, unsigned int delta) {
    return __shfl_down_sync(0xffffffff, val, delta, 32);
}

// ✅ Specialization for __nv_bfloat16 (uses intrinsic shuffle)
__device__ inline __nv_bfloat16 shfl_down(__nv_bfloat16 val, unsigned int delta) {
    return __shfl_down_sync(0xffffffff, val, delta, 32);
}

// ═══════════════════════════════════════════════════════════
// WARP-LEVEL REDUCTION (USES GPU INTRINSICS)
// ═══════════════════════════════════════════════════════════

template<typename T, template<typename> class OpType>
__device__ T warp_reduce(T val, OpType<T> op) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = shfl_down(val, offset);
        val = op.reduce(val, other);
    }
    return val;
}

// ═══════════════════════════════════════════════════════════
// BLOCK-LEVEL REDUCTION
// ═══════════════════════════════════════════════════════════

template<typename AccT, typename T, template<typename> class OpType>
__device__ AccT block_reduce(AccT val, AccT* shared, OpType<T> op) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp reduction
    OpType<AccT> acc_op;
    val = warp_reduce(val, acc_op);

    // Write warp results
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : acc_op.identity();
        val = warp_reduce(val, acc_op);
    }

    return val;
}

// ═══════════════════════════════════════════════════════════
// MAIN REDUCTION KERNEL (NATIVE CUDA TYPES)
// ═══════════════════════════════════════════════════════════

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
    
    // Determine accumulator type
    constexpr bool is_half = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
    constexpr bool is_integer_sum = std::is_integral_v<T> && std::is_same_v<OpType<T>, detail::SumOp<T>>;
    constexpr bool is_integer_product = std::is_integral_v<T> && std::is_same_v<OpType<T>, detail::ProductOp<T>>;

    using AccumulatorType = typename std::conditional_t<
        is_integer_sum || is_integer_product,
        int64_t,
        typename std::conditional_t<is_half, float, T>
    >;

    extern __shared__ char shared_mem[];
    AccumulatorType* shared = reinterpret_cast<AccumulatorType*>(shared_mem);

    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        
        // Initialize accumulator
        AccumulatorType accumulator;
        if constexpr (is_integer_sum) {
            accumulator = 0LL;
        } else if constexpr (is_integer_product) {
            accumulator = 1LL;
        } else if constexpr (is_half) {
            accumulator = to_float(op.identity());
        } else {
            accumulator = op.identity();
        }

        // Calculate output coordinates
        int64_t out_coords[10];
        int64_t temp = output_index;
        for (int d = num_reduced_dims - 1; d >= 0; --d) {
            out_coords[d] = temp % output_dims[d];
            temp /= output_dims[d];
        }

        // ✅ ACCUMULATION LOOP (USES GPU INTRINSICS FOR HALF TYPES)
        for (int64_t i = threadIdx.x; i < reduced_count; i += blockDim.x) {
            // Calculate input coordinates
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
                    full_input_coords[dim] = rank_preserved ? out_coords[dim] : out_coords[out_coord_idx];
                    if (!rank_preserved) out_coord_idx++;
                }
            }

            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }

            T input_value = input_data[input_lin_idx];

            // ✅ ACCUMULATE USING GPU INTRINSICS
            if constexpr (is_half) {
                float val_f = to_float(input_value);
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
            int lane = threadIdx.x % 32;
            int wid = threadIdx.x / 32;

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                int64_t other = shfl_down(accumulator, offset);
                if constexpr (is_integer_sum) accumulator += other;
                else accumulator *= other;
            }

            if (lane == 0) shared[wid] = accumulator;
            __syncthreads();

            if (wid == 0) {
                accumulator = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 
                             (is_integer_sum ? 0LL : 1LL);
                
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    int64_t other = shfl_down(accumulator, offset);
                    if constexpr (is_integer_sum) accumulator += other;
                    else accumulator *= other;
                }
            }

            if (threadIdx.x == 0) {
                output_data[output_index] = static_cast<OutputT>(accumulator);
            }
        } else {
            AccumulatorType final_val = block_reduce<AccumulatorType, T, OpType>(accumulator, shared, op);

            if (threadIdx.x == 0) {
                if constexpr (is_half) {
                    output_data[output_index] = from_float<OutputT>(final_val);
                } else {
                    output_data[output_index] = static_cast<OutputT>(final_val);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// INDEX REDUCTION KERNEL (NATIVE CUDA TYPES)
// ═══════════════════════════════════════════════════════════

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
                    full_input_coords[dim] = rank_preserved ? out_coords[dim] : out_coords[out_coord_idx];
                    if (!rank_preserved) out_coord_idx++;
                }
            }

            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }

            T input_value = input_data[input_lin_idx];
            ValueIndexType current = {input_value, i};
            accumulator = op.reduce(accumulator, current);
        }

        // Warp reduction
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            ValueIndexType other;
            other.value = shfl_down(accumulator.value, offset);
            other.index = shfl_down(accumulator.index, offset);
            accumulator = op.reduce(accumulator, other);
        }

        if (lane == 0) shared[wid] = accumulator;
        __syncthreads();

        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared[lane] : op.identity();

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ValueIndexType other;
                other.value = shfl_down(accumulator.value, offset);
                other.index = shfl_down(accumulator.index, offset);
                accumulator = op.reduce(accumulator, other);
            }
        }

        if (threadIdx.x == 0) {
            output_data[output_index] = accumulator.index;
        }
    }
}

// ═══════════════════════════════════════════════════════════
// MEAN REDUCTION KERNEL (NATIVE CUDA TYPES + INTRINSICS)
// ═══════════════════════════════════════════════════════════

template<typename T, typename OutputT, template<typename> class SumOpType>
__global__ void reduce_mean_kernel(
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
    constexpr bool is_nan_aware = std::is_same_v<SumOpType<T>, detail::NanSumOp<T>>;
    constexpr bool is_half = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;

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

        // ✅ ACCUMULATION WITH GPU INTRINSICS
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
                    full_input_coords[dim] = rank_preserved ? out_coords[dim] : out_coords[out_coord_idx];
                    if (!rank_preserved) out_coord_idx++;
                }
            }

            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }

            T input_value = input_data[input_lin_idx];

            // ✅ CONVERT USING GPU INTRINSICS
            double val_d;
            if constexpr (is_half) {
                val_d = static_cast<double>(to_float(input_value));
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

        // Warp reduction
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            double other_acc = shfl_down(accumulator, offset);
            accumulator += other_acc;
            
            if constexpr (is_nan_aware) {
                int64_t other_count = shfl_down(valid_count, offset);
                valid_count += other_count;
            }
        }

        if (lane == 0) {
            shared_acc[wid] = accumulator;
            if constexpr (is_nan_aware) shared_count[wid] = valid_count;
        }
        __syncthreads();

        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared_acc[lane] : 0.0;
            if constexpr (is_nan_aware) {
                valid_count = (threadIdx.x < blockDim.x / 32) ? shared_count[lane] : 0;
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                double other_acc = shfl_down(accumulator, offset);
                accumulator += other_acc;
                
                if constexpr (is_nan_aware) {
                    int64_t other_count = shfl_down(valid_count, offset);
                    valid_count += other_count;
                }
            }
        }

        if (threadIdx.x == 0) {
            double mean_val;
            
            if constexpr (is_nan_aware) {
                mean_val = (valid_count == 0) ? 
                    __longlong_as_double(0x7ff8000000000000ULL) : 
                    accumulator / static_cast<double>(valid_count);
            } else {
                mean_val = accumulator / static_cast<double>(reduced_count);
            }

            // ✅ CONVERT BACK USING GPU INTRINSICS
            if constexpr (is_half) {
                output_data[output_index] = from_float<OutputT>(static_cast<float>(mean_val));
            } else {
                output_data[output_index] = static_cast<OutputT>(mean_val);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
// VARIANCE REDUCTION KERNEL (FIXED - Now accepts separate MeanT type)
// ═══════════════════════════════════════════════════════════

template<typename T, typename MeanT, typename OutputT, template<typename> class VarianceOpType>
__global__ void reduce_variance_kernel(
    const T* __restrict__ input_data,
    const MeanT* __restrict__ mean_data,  // ✅ SEPARATE TYPE for mean!
    OutputT* __restrict__ output_data,
    const int64_t* __restrict__ input_dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_dims,
    const int64_t* __restrict__ normalized_axes,
    const int64_t* __restrict__ reduced_dims,
    int64_t num_slices,
    int64_t reduced_count,
    int64_t correction, //Bessel's correction parameter
    int ndim,
    int num_axes,
    int num_reduced_dims,
    bool rank_preserved)
{
    // ✅ Determine if this is NaN-aware variance
    constexpr bool is_nan_aware = std::is_same_v<VarianceOpType<T>, detail::NanVarianceOp<T>>;
    
    // ✅ Use MeanT for accumulation (matches mean tensor type)
    using AccT = typename std::conditional<
        std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
        float,
        MeanT  // ✅ Use mean type for accumulator
    >::type;
    
    extern __shared__ char shared_mem[];
    AccT* shared_acc = reinterpret_cast<AccT*>(shared_mem);
    int64_t* shared_count = reinterpret_cast<int64_t*>(shared_acc + blockDim.x / 32);
    
    for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
        
        // Calculate output coordinates
        int64_t out_coords[10];
        {
            int64_t temp = output_index;
            int output_ndim = rank_preserved ? ndim : 0;
            
            if (!rank_preserved) {
                for (int dim = 0; dim < ndim; ++dim) {
                    bool is_reduced = false;
                    for (int ax = 0; ax < num_axes; ++ax) {
                        if (normalized_axes[ax] == dim) {
                            is_reduced = true;
                            break;
                        }
                    }
                    if (!is_reduced) output_ndim++;
                }
            }
            
            for (int d = output_ndim - 1; d >= 0; --d) {
                out_coords[d] = temp % output_dims[d];
                temp /= output_dims[d];
            }
        }

        // Map output coords to mean tensor coords (keepdim=true → reduced dims = 1)
        int64_t mean_coords[10];
        int out_idx = 0;

        for (int dim = 0; dim < ndim; ++dim) {
            bool is_reduced = false;
            for (int ax = 0; ax < num_axes; ++ax) {
                if (normalized_axes[ax] == dim) {
                    is_reduced = true;
                    break;
                }
            }
            
            if (is_reduced) {
                mean_coords[dim] = 0;
            } else {
                if (rank_preserved) {
                    mean_coords[dim] = out_coords[dim];
                } else {
                    mean_coords[dim] = out_coords[out_idx];
                    out_idx++;
                }
            }
        }

        // ✅ Compute linear index for mean tensor
        int64_t mean_shape[10];
        for (int d = 0; d < ndim; ++d) {
            bool is_reduced = false;
            for (int ax = 0; ax < num_axes; ++ax) {
                if (normalized_axes[ax] == d) {
                    is_reduced = true;
                    break;
                }
            }
            mean_shape[d] = is_reduced ? 1 : input_dims[d];
        }
        
        int64_t mean_index = 0;
        int64_t mean_stride = 1;
        
        for (int d = ndim - 1; d >= 0; --d) {
            mean_index += mean_coords[d] * mean_stride;
            mean_stride *= mean_shape[d];
        }
        
        // ✅ Get mean value - now correctly typed!
        AccT mean_val;
        if constexpr (std::is_same_v<MeanT, __half> || std::is_same_v<MeanT, __nv_bfloat16>) {
            mean_val = to_float(mean_data[mean_index]);
        } else {
            mean_val = static_cast<AccT>(mean_data[mean_index]);
        }
        
        AccT accumulator = 0;
        int64_t valid_count = 0;
        
        // Accumulate squared deviations
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
                    full_input_coords[dim] = rank_preserved ? out_coords[dim] : out_coords[out_coord_idx];
                    if (!rank_preserved) out_coord_idx++;
                }
            }
            
            int64_t input_lin_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                input_lin_idx += full_input_coords[d] * input_strides[d];
            }
            
            T input_value = input_data[input_lin_idx];
            
            // ✅ Convert input to AccT (which matches mean type)
            AccT val;
            if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
                val = to_float(input_value);
            } else {
                val = static_cast<AccT>(input_value);
            }
            
            if constexpr (is_nan_aware) {
                if (!isnan(val)) {
                    AccT diff = val - mean_val;
                    accumulator += diff * diff;
                    valid_count++;
                }
            } else {
                if (isnan(val) || isnan(mean_val)) {
                    accumulator = nanf("");
                } else {
                    AccT diff = val - mean_val;
                    accumulator += diff * diff;
                }
            }
        }
        
        // Block reduction
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            AccT other_acc = shfl_down(accumulator, offset);
            accumulator += other_acc;
            
            if constexpr (is_nan_aware) {
                int64_t other_count = shfl_down(valid_count, offset);
                valid_count += other_count;
            }
        }
        
        if (lane == 0) {
            shared_acc[wid] = accumulator;
            if constexpr (is_nan_aware) shared_count[wid] = valid_count;
        }
        __syncthreads();
        
        if (wid == 0) {
            accumulator = (threadIdx.x < blockDim.x / 32) ? shared_acc[lane] : AccT(0);
            if constexpr (is_nan_aware) {
                valid_count = (threadIdx.x < blockDim.x / 32) ? shared_count[lane] : 0;
            }
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                AccT other_acc = shfl_down(accumulator, offset);
                accumulator += other_acc;
                
                if constexpr (is_nan_aware) {
                    int64_t other_count = shfl_down(valid_count, offset);
                    valid_count += other_count;
                }
            }
        }
        
        // Final division and output
        if (threadIdx.x == 0) {
            int64_t divisor;
            if constexpr (is_nan_aware) {
                divisor = valid_count - correction;
            } else {
                divisor = reduced_count - correction;
            }
            
            if (isnan(accumulator)) {
                if constexpr (std::is_same_v<OutputT, __half>) {
                    output_data[output_index] = __float2half(nanf(""));
                } else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
                    output_data[output_index] = __float2bfloat16(nanf(""));
                } else {
                    output_data[output_index] = static_cast<OutputT>(nanf(""));
                }
            } else if (divisor <= 0) {
                // Insufficient data - return NaN
                if constexpr (std::is_same_v<OutputT, __half>) {
                    output_data[output_index] = __float2half(nanf(""));
                } else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
                    output_data[output_index] = __float2bfloat16(nanf(""));
                } else {
                    output_data[output_index] = static_cast<OutputT>(nanf(""));
                }
            } else {
                // Compute variance
                AccT variance = accumulator / static_cast<AccT>(divisor);
                
                if constexpr (std::is_same_v<OutputT, __half> || std::is_same_v<OutputT, __nv_bfloat16>) {
                    output_data[output_index] = from_float<OutputT>(static_cast<float>(variance));
                } else {
                    output_data[output_index] = static_cast<OutputT>(variance);
                }
            }
        }
    }
}
} // namespace cuda
} // namespace OwnTensor

#endif // REDUCTION_KERNELS_CUH