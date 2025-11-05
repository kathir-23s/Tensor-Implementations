// src/UnaryOps/ReductionImplGPU.cu - FIXED: Added missing includes
#include "ops/helpers/ReductionKernels.cuh"
#include "ops/helpers/ReductionImplGPU.h"
#include "ops/helpers/ReductionUtils.h"  // ← CRITICAL: Adds calculate_output_shape, calculate_reduced_count
#include "core/Tensor.h"                  // ← CRITICAL: Adds Tensor, Shape, TensorOptions
#include <cuda_runtime.h>

// ✅ CRITICAL: Use OwnTensor's custom types (NOT native CUDA types)
#include "dtype/Types.h"

namespace OwnTensor {
namespace detail {

#ifdef WITH_CUDA

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

// =================================================================
// GPU VALUE REDUCTION DISPATCHER
// =================================================================

template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input, 
                               const std::vector<int64_t>& normalized_axes, 
                               bool keepdim, cudaStream_t stream) //✨✨✨
{
    // Calculate output shape
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    // Determine correct output dtype based on input type
    Dtype output_dtype;
    if constexpr (std::is_integral_v<T>) {
        output_dtype = Dtype::Int64;
    } else {
        output_dtype = input.dtype();
    }
    
    // Create output tensor on GPU
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype)
        .with_device(input.device())
        .with_req_grad(false));
    
    // Setup
    const T* input_data = input.data<T>();
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
    DeviceArray d_input_dims(input_dims, stream);//✨✨✨
    DeviceArray d_input_strides(input_strides, stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims, stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes, stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims, stream);//✨✨✨
    
    // Kernel configuration
    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    // Calculate shared memory for accumulator type
    size_t shared_mem_size;
    if constexpr (std::is_integral_v<T>) {
        shared_mem_size = (threads_per_block / 32) * sizeof(int64_t);
    } else {
        shared_mem_size = (threads_per_block / 32) * sizeof(T);
    }
    
    // ✅ FIXED: Proper std::conditional syntax with angle brackets
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        int64_t,
        T
    >::type;
    
    OutputCppT* output_data = output.data<OutputCppT>();
    
    // Launch kernel with correct template parameters
    cuda::reduce_kernel<T, OutputCppT, OpType><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(//✨✨✨
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
    
    //cudaDeviceSynchronize();//✨✨✨
    
    return output;
}

// =================================================================
// GPU INDEX REDUCTION DISPATCHER
// =================================================================

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input, 
                                     const std::vector<int64_t>& normalized_axes, 
                                     bool keepdim, cudaStream_t stream) //✨✨✨
{
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(Dtype::Int64)
        .with_device(input.device())
        .with_req_grad(false));
    
    const T* input_data = input.data<T>();
    int64_t* output_data = output.data<int64_t>();
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
    
    DeviceArray d_input_dims(input_dims, stream);//✨✨✨
    DeviceArray d_input_strides(input_strides, stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims, stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes, stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims, stream);//✨✨✨
    
    int threads_per_block = 256;
    int num_blocks = num_slices;
    size_t shared_mem_size = (threads_per_block / 32) * sizeof(detail::ValueIndex<T>);
    
    cuda::reduce_index_kernel<T, OpType><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(//✨✨✨
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
    
    //cudaDeviceSynchronize();//✨✨✨
    
    return output;
}

// =================================================================
// GPU MEAN REDUCTION DISPATCHER
// =================================================================

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
    
    // Mean output: Int64 → Float64, others stay same type
    Dtype output_dtype;
    if constexpr (std::is_integral_v<T>) {
        output_dtype = Dtype::Float64;
    } else {
        output_dtype = input.dtype();
    }
    
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype)
        .with_device(input.device())
        .with_req_grad(false));
    
    const T* input_data = input.data<T>();
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
    
    DeviceArray d_input_dims(input_dims, stream);//✨✨✨
    DeviceArray d_input_strides(input_strides, stream);//✨✨✨
    DeviceArray d_output_dims(output_shape.dims, stream);//✨✨✨
    DeviceArray d_normalized_axes(normalized_axes, stream);//✨✨✨
    DeviceArray d_reduced_dims(reduced_dims, stream);//✨✨✨
    
    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    int num_warps = (threads_per_block + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double) + num_warps * sizeof(int64_t);
    
    // ✅ FIXED: Proper std::conditional syntax with angle brackets
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    OutputCppT* output_data = output.data<OutputCppT>();
    
    cuda::reduce_mean_kernel<T, SumOpType><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(//✨✨✨
        input_data,
        reinterpret_cast<T*>(output_data),
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
    
    //cudaDeviceSynchronize();//✨✨✨
    
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

// int32_t (int) - Basic operations only
template Tensor dispatch_reduction_gpu<int32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨

// int64_t (long) - Basic operations only
template Tensor dispatch_reduction_gpu<int64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨

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

#endif // WITH_CUDA

} // namespace detail
} // namespace OwnTensor