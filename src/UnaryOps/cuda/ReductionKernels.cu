// src/UnaryOps/ReductionKernels.cu - EXPLICIT INSTANTIATIONS WITH CUSTOM STRUCTS
#include "ops/helpers/ReductionKernels.cuh"
#include "dtype/Dtype.h"  
#include "dtype/DtypeTraits.h"
#include "ops/helpers/ReductionOps.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <limits>

namespace OwnTensor {
namespace cuda {

// =================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS - Using Custom Structs
// =================================================================

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

// ===========================================================
// INTEGER TYPES - BASIC OPERATIONS ONLY (NO NaN)
// ===========================================================

// int16_t (short) - Basic operations only
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(int16_t, int64_t, detail::MaxOp)
INSTANTIATE_INDEX_KERNEL(int16_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(int16_t, detail::ArgMaxOp)
INSTANTIATE_MEAN_KERNEL(int16_t, detail::SumOp)

// int32_t (int) - Basic operations only
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(int32_t, int64_t, detail::MaxOp)
INSTANTIATE_INDEX_KERNEL(int32_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(int32_t, detail::ArgMaxOp)
INSTANTIATE_MEAN_KERNEL(int32_t, detail::SumOp)

// int64_t (long) - Basic operations only
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(int64_t, int64_t, detail::MaxOp)
INSTANTIATE_INDEX_KERNEL(int64_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(int64_t, detail::ArgMaxOp)
INSTANTIATE_MEAN_KERNEL(int64_t, detail::SumOp)

// ===========================================================
// FLOATING POINT TYPES - Using Custom Structs
// ===========================================================

// float16_t (custom struct)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(float16_t, float16_t, detail::NanMaxOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(float16_t, detail::NanArgMaxOp)
INSTANTIATE_MEAN_KERNEL(float16_t, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(float16_t, detail::NanSumOp)

// bfloat16_t (custom struct)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(bfloat16_t, bfloat16_t, detail::NanMaxOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(bfloat16_t, detail::NanArgMaxOp)
INSTANTIATE_MEAN_KERNEL(bfloat16_t, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(bfloat16_t, detail::NanSumOp)

// float - All operations
INSTANTIATE_REDUCE_KERNEL(float, float, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(float, float, detail::NanMaxOp)
INSTANTIATE_INDEX_KERNEL(float, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(float, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(float, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(float, detail::NanArgMaxOp)
INSTANTIATE_MEAN_KERNEL(float, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(float, detail::NanSumOp)

// double - All operations
INSTANTIATE_REDUCE_KERNEL(double, double, detail::SumOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::ProductOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::MinOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::MaxOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanSumOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanProductOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanMinOp)
INSTANTIATE_REDUCE_KERNEL(double, double, detail::NanMaxOp)
INSTANTIATE_INDEX_KERNEL(double, detail::ArgMinOp)
INSTANTIATE_INDEX_KERNEL(double, detail::ArgMaxOp)
INSTANTIATE_INDEX_KERNEL(double, detail::NanArgMinOp)
INSTANTIATE_INDEX_KERNEL(double, detail::NanArgMaxOp)
INSTANTIATE_MEAN_KERNEL(double, detail::SumOp)
INSTANTIATE_MEAN_KERNEL(double, detail::NanSumOp)

} // namespace cuda
} // namespace OwnTensor