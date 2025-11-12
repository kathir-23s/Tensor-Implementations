// src/UnaryOps/ReductionImplGPU.cu - FIXED: Added type conversion layer
#include "ops/helpers/ReductionKernels.cuh"
#include "ops/helpers/ReductionUtils.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>

// ✅ CRITICAL: Include both custom structs AND native CUDA types
#include "dtype/Types.h"        // Custom structs (float16_t, bfloat16_t)
#include <cuda_fp16.h>          // Native CUDA types (__half, __nv_bfloat16)
#include <cuda_bf16.h>

namespace OwnTensor {
namespace detail {

#ifdef WITH_CUDA

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION TRAITS (Custom Struct → Native CUDA Type)
// ═══════════════════════════════════════════════════════════

template<typename T> struct ToCudaNative { using type = T; };
template<> struct ToCudaNative<float16_t> { using type = __half; };
template<> struct ToCudaNative<bfloat16_t> { using type = __nv_bfloat16; };

template<typename T>
using CudaNativeType = typename ToCudaNative<T>::type;

// =================================================================
// GPU DEVICE MEMORY HELPER
// =================================================================

class DeviceArray {
public:
    int64_t* ptr;
    
    DeviceArray(const std::vector<int64_t>& host_data, cudaStream_t stream) {//✨✨✨
        size_t bytes = host_data.size() * sizeof(int64_t);
        cudaMalloc(&ptr, bytes);
        cudaMemcpyAsync(ptr, host_data.data(), bytes, cudaMemcpyHostToDevice, stream);//✨✨✨
    }
    
    ~DeviceArray() {
        if (ptr) cudaFree(ptr);
    }
    
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// ═══════════════════════════════════════════════════════════
// GPU VALUE REDUCTION DISPATCHER (WITH TYPE CONVERSION)
// ═══════════════════════════════════════════════════════════

template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input, 
                               const std::vector<int64_t>& normalized_axes, 
                               bool keepdim,cudaStream_t stream) //✨✨✨
{
    // Calculate output shape
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    // Determine output dtype
    Dtype output_dtype;
    if constexpr (std::is_same_v<T, bool>) {
        output_dtype = Dtype::Bool;  // ✅ Boolean operations return Bool
    } else if constexpr (std::is_integral_v<T>) {
        output_dtype = Dtype::Int64;
    } else {
        output_dtype = input.dtype();
    }
    
    // Create output tensor
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype)
        .with_device(input.device())
        .with_req_grad(input.requires_grad()));
    
    // Setup metadata
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    const int64_t num_slices = output.numel();
    const int64_t reduced_count = calculate_reduced_count(input_dims, normalized_axes);
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    if (reduced_count == 0) {
        throw std::runtime_error("GPU Reduction error: reduced count is zero");
    }
    
    // Calculate reduced_dims
    std::vector<int64_t> reduced_dims;
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                         != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    
    // Transfer metadata to device
    DeviceArray d_input_dims(input_dims,stream);//✨✨✨
    DeviceArray d_input_strides(input_strides,stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims,stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes,stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims,stream);//✨✨✨
    
    // Kernel configuration
    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    // ✅ TYPE CONVERSION: Custom struct → Native CUDA type
    using CudaT = CudaNativeType<T>;
    
    // Calculate shared memory for accumulator type
    size_t shared_mem_size;
    if constexpr (std::is_integral_v<T>) {
        shared_mem_size = (threads_per_block / 32) * sizeof(int64_t);
    } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
        shared_mem_size = (threads_per_block / 32) * sizeof(float);
    } else {
        shared_mem_size = (threads_per_block / 32) * sizeof(T);
    }
    
    // ✅ FIXED: Proper output type selection
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        int64_t,
        T
    >::type;
    
    // ✅ CRITICAL: Cast pointers to NATIVE CUDA types
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input.data<T>());
    
    using OutputCudaT = CudaNativeType<OutputCppT>;
    OutputCudaT* output_data = reinterpret_cast<OutputCudaT*>(output.data<OutputCppT>());
    
    // ✅ Launch kernel with NATIVE CUDA types
    cuda::reduce_kernel<CudaT, OutputCudaT, OpType><<<num_blocks, threads_per_block, shared_mem_size,stream>>>(
        input_data,
        output_data,
        d_input_dims.ptr,
        d_input_strides.ptr,
        d_output_dims.ptr,
        d_normalized_axes.ptr,
        d_reduced_dims.ptr,
        num_slices,
        reduced_count,
        input_dims.size(),
        normalized_axes.size(),
        reduced_dims.size(),
        rank_preserved
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + 
                               cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    return output;
}

// ═══════════════════════════════════════════════════════════
// GPU INDEX REDUCTION DISPATCHER (WITH TYPE CONVERSION)
// ═══════════════════════════════════════════════════════════

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input, 
                                     const std::vector<int64_t>& normalized_axes, 
                                     bool keepdim,cudaStream_t stream) //✨✨✨
{
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(Dtype::Int64)
        .with_device(input.device())
        .with_req_grad(input.requires_grad()));
    
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    const int64_t num_slices = output.numel();
    const int64_t reduced_count = calculate_reduced_count(input_dims, normalized_axes);
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    if (reduced_count == 0) {
        throw std::runtime_error("GPU Index Reduction error: reduced count is zero");
    }
    
    std::vector<int64_t> reduced_dims;
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                         != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    
    DeviceArray d_input_dims(input_dims,stream); //✨✨✨
    DeviceArray d_input_strides(input_strides,stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims,stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes,stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims,stream);//✨✨✨

    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    // ✅ TYPE CONVERSION
    using CudaT = CudaNativeType<T>;
    size_t shared_mem_size = (threads_per_block / 32) * sizeof(detail::ValueIndex<CudaT>);
    
    // ✅ Cast pointers to native CUDA types
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input.data<T>());
    int64_t* output_data = output.data<int64_t>();
    
    cuda::reduce_index_kernel<CudaT, OpType><<<num_blocks, threads_per_block, shared_mem_size,stream>>>(
        input_data,
        output_data,
        d_input_dims.ptr,
        d_input_strides.ptr,
        d_output_dims.ptr,
        d_normalized_axes.ptr,
        d_reduced_dims.ptr,
        num_slices,
        reduced_count,
        input_dims.size(),
        normalized_axes.size(),
        reduced_dims.size(),
        rank_preserved
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA index kernel launch failed: ") + 
                               cudaGetErrorString(err));
    }
    
    //cudaDeviceSynchronize();
    
    return output;
}

// ═══════════════════════════════════════════════════════════
// GPU MEAN REDUCTION DISPATCHER (WITH TYPE CONVERSION)
// ═══════════════════════════════════════════════════════════

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor& input, 
                         const std::vector<int64_t>& normalized_axes, 
                         bool keepdim, cudaStream_t stream) //✨✨✨ 
{
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);
    if (reduced_count == 0) {
        throw std::runtime_error("Cannot compute mean: reduced count is zero.");
    }
    
    // Mean output dtype
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
    
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    const int64_t num_slices = output.numel();
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    std::vector<int64_t> reduced_dims;
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                         != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    
    DeviceArray d_input_dims(input_dims,stream); //✨✨✨
    DeviceArray d_input_strides(input_strides,stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims,stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes,stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims,stream);//✨✨✨

    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    int num_warps = (threads_per_block + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double) + num_warps * sizeof(int64_t);
    
    // ✅ TYPE CONVERSION
    using CudaT = CudaNativeType<T>;
    
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    using OutputCudaT = CudaNativeType<OutputCppT>;
    
    // ✅ Cast pointers to native CUDA types
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input.data<T>());
    OutputCudaT* output_data = reinterpret_cast<OutputCudaT*>(output.data<OutputCppT>());
    
    cuda::reduce_mean_kernel<CudaT, OutputCudaT, SumOpType><<<num_blocks, threads_per_block, shared_mem_size,stream>>>(
        input_data,
        output_data,
        d_input_dims.ptr,
        d_input_strides.ptr,
        d_output_dims.ptr,
        d_normalized_axes.ptr,
        d_reduced_dims.ptr,
        num_slices,
        reduced_count,
        input_dims.size(),
        normalized_axes.size(),
        reduced_dims.size(),
        rank_preserved
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA mean kernel launch failed: ") + 
                               cudaGetErrorString(err));
    }
    
    //cudaDeviceSynchronize();
    
    return output;
}
// ═══════════════════════════════════════════════════════════
// GPU VARIANCE REDUCTION DISPATCHER (COMPLETE IMPLEMENTATION)
//═══════════════════════════════════════════════════════════

template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_gpu(const Tensor& input, 
                             const std::vector<int64_t>& normalized_axes, 
                             bool keepdim,
                             int64_t correction, cudaStream_t stream) //✨✨✨ 
                             {
    constexpr bool is_nan_aware = std::is_same_v<VarianceOpType<T>, NanVarianceOp<T>>;
    
    // ✅ STEP 1: Compute mean on GPU (always keepdim=true for variance)
    Tensor mean_tensor = is_nan_aware
        ? dispatch_mean_gpu<T, NanSumOp>(input, normalized_axes, true, stream)
        : dispatch_mean_gpu<T, SumOp>(input, normalized_axes, true, stream);

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
    
    // Create output tensor
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype)
        .with_device(input.device())
        .with_req_grad(input.requires_grad()));
    
    // Setup metadata
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    const int64_t num_slices = output.numel();
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    // Calculate reduced_dims
    std::vector<int64_t> reduced_dims;
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                         != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    
    // ✅ STEP 3: Transfer metadata to device
    DeviceArray d_input_dims(input_dims, stream);//✨✨✨
    DeviceArray d_input_strides(input_strides, stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims, stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes, stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims, stream);//✨✨✨

    // ✅ STEP 4: Kernel configuration
    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    // Type conversion for input
    using CudaT = CudaNativeType<T>;
    
    // ✅ CRITICAL FIX: Determine the ACTUAL mean tensor type
    // For integers: mean_tensor has dtype Float64, so mean is stored as double
    // For floats: mean_tensor has same dtype as input
    using MeanCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,  // Integer inputs → Float64 mean
        T        // Float inputs → same type mean
    >::type;
    
    using MeanCudaT = CudaNativeType<MeanCppT>;
    
    // Output type
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    using OutputCudaT = CudaNativeType<OutputCppT>;
    
    // Calculate shared memory size
    size_t shared_mem_size;
    if constexpr (std::is_integral_v<T>) {
        shared_mem_size = (threads_per_block / 32) * sizeof(double);
    } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
        shared_mem_size = (threads_per_block / 32) * sizeof(float);
    } else {
        shared_mem_size = (threads_per_block / 32) * sizeof(T);
    }
    
    // ✅ STEP 5: Cast pointers to CORRECT native CUDA types
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input.data<T>());
    
    // ✅ FIX: Use the ACTUAL mean tensor type (not the input type!)
    const MeanCudaT* mean_data = reinterpret_cast<const MeanCudaT*>(mean_tensor.data<MeanCppT>());
    
    OutputCudaT* output_data = reinterpret_cast<OutputCudaT*>(output.data<OutputCppT>());
    
    // ✅ STEP 6: LAUNCH THE VARIANCE KERNEL
    cuda::reduce_variance_kernel<CudaT, MeanCudaT, OutputCudaT, VarianceOpType>
        <<<num_blocks, threads_per_block, shared_mem_size, stream>>>(//✨✨✨
        input_data,
        mean_data,           // Pre-computed mean (CORRECT TYPE!)
        output_data,
        d_input_dims.ptr,
        d_input_strides.ptr,
        d_output_dims.ptr,
        d_normalized_axes.ptr,
        d_reduced_dims.ptr,
        num_slices,
        reduced_count,
        correction,          // Bessel's correction parameter
        input_dims.size(),
        normalized_axes.size(),
        reduced_dims.size(),
        rank_preserved
    );
    
    // ✅ STEP 7: Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA variance kernel launch failed: ") + 
                               cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    return output;
}

// =================================================================
// ✅ EXPLICIT TEMPLATE INSTANTIATIONS - Using Custom Structs
// =================================================================

// ===========================================================
// INTEGER TYPES - BASIC OPERATIONS ONLY (NO NaN)
// ===========================================================

// int16_t (short) - Basic operations only
template Tensor dispatch_reduction_gpu<int16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int16_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨ 
template Tensor dispatch_variance_gpu<int16_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨   

// int32_t (int) - Basic operations only
template Tensor dispatch_reduction_gpu<int32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int32_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<int32_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// int64_t (long) - Basic operations only
template Tensor dispatch_reduction_gpu<int64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int64_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<int64_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// ===========================================================
// ✅ FLOATING POINT - Using CUSTOM STRUCTS (NOT __half/__nv_bfloat16)
// ===========================================================

// float16_t (custom struct)
template Tensor dispatch_reduction_gpu<float16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<float16_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction  , cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<float16_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
// bfloat16_t (custom struct)
template Tensor dispatch_reduction_gpu<bfloat16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<bfloat16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<bfloat16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<bfloat16_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<bfloat16_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// float - All operations
template Tensor dispatch_reduction_gpu<float, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<float, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<float, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
// double - All operations
template Tensor dispatch_reduction_gpu<double, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_mean_gpu<double, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_mean_gpu<double, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_variance_gpu<double, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<double, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

//Boolean type - Basic operations only
template Tensor dispatch_mean_gpu<bool, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
// template Tensor dispatch_variance_gpu<bool, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
// template Tensor dispatch_variance_gpu<bool, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_reduction_gpu<bool, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bool, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bool, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bool, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bool, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bool, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨   
// Add to the Bool section at the end of the file

// Boolean-specific reductions
template Tensor dispatch_reduction_gpu<bool, AllOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<bool, AnyOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
#endif // WITH_CUDA

} // namespace detail
} // namespace OwnTensor