// src/UnaryOps/ReductionImplGPU.cu - FIXED VERSION
#include "ReductionKernels.cuh"
#include "ops/helpers/ReductionImplGPU.h"
#include <cuda_runtime.h>

namespace OwnTensor {
namespace detail {

#ifdef WITH_CUDA

// =================================================================
// GPU DEVICE MEMORY HELPER
// =================================================================

class DeviceArray {
public:
    int64_t* ptr;
    
    DeviceArray(const std::vector<int64_t>& host_data) {
        size_t bytes = host_data.size() * sizeof(int64_t);
        cudaMalloc(&ptr, bytes);
        cudaMemcpy(ptr, host_data.data(), bytes, cudaMemcpyHostToDevice);
    }
    
    ~DeviceArray() {
        if (ptr) cudaFree(ptr);
    }
    
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// =================================================================
// GPU VALUE REDUCTION DISPATCHER - FIXED OUTPUT TYPE
// =================================================================

template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input, 
                               const std::vector<int64_t>& normalized_axes, 
                               bool keepdim) 
{
    // Calculate output shape
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    // ✅ FIXED: Determine correct output dtype based on input type
    Dtype output_dtype;
    if constexpr (std::is_integral_v<T>) {
        output_dtype = Dtype::Int64;  // Integers widen to Int64
    } else {
        output_dtype = input.dtype();  // Floats stay same type
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
    DeviceArray d_input_dims(input_dims);
    DeviceArray d_input_strides(input_strides);
    DeviceArray d_output_dims(output_shape.dims);
    DeviceArray d_normalized_axes(normalized_axes);
    DeviceArray d_reduced_dims(reduced_dims);
    
    // Kernel configuration
    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    // ✅ FIXED: Calculate shared memory for accumulator type
    size_t shared_mem_size;
    if constexpr (std::is_integral_v<T>) {
        shared_mem_size = (threads_per_block / 32) * sizeof(int64_t);  // int64_t accumulation
    } else {
        shared_mem_size = (threads_per_block / 32) * sizeof(T);  // T accumulation
    }
    
    // ✅ FIXED: Determine correct output C++ type
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        int64_t,  // Integers â†' int64_t
        T         // Floats â†' T
    >::type;
    
    OutputCppT* output_data = output.data<OutputCppT>();
    
    // ✅ FIXED: Launch kernel with correct template parameters
    cuda::reduce_kernel<T, OutputCppT, OpType><<<num_blocks, threads_per_block, shared_mem_size>>>(
        input_data,
        output_data,  // Now correctly typed!
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

// =================================================================
// GPU INDEX REDUCTION DISPATCHER (argmin/argmax) - UNCHANGED
// =================================================================

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input, 
                                     const std::vector<int64_t>& normalized_axes, 
                                     bool keepdim) 
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
    
    DeviceArray d_input_dims(input_dims);
    DeviceArray d_input_strides(input_strides);
    DeviceArray d_output_dims(output_shape.dims);
    DeviceArray d_normalized_axes(normalized_axes);
    DeviceArray d_reduced_dims(reduced_dims);
    
    int threads_per_block = 256;
    int num_blocks = num_slices;
    size_t shared_mem_size = (threads_per_block / 32) * sizeof(detail::ValueIndex<T>);
    
    cuda::reduce_index_kernel<T, OpType><<<num_blocks, threads_per_block, shared_mem_size>>>(
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
    
    cudaDeviceSynchronize();
    
    return output;
}

// =================================================================
// GPU MEAN REDUCTION DISPATCHER - FIXED SHARED MEMORY
// =================================================================

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor& input, 
                         const std::vector<int64_t>& normalized_axes, 
                         bool keepdim) 
{
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    
    int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);
    if (reduced_count == 0) {
        throw std::runtime_error("Cannot compute mean: reduced count is zero.");
    }
    
    // Mean output: Int64 â†' Float64, others stay same type
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
    
    DeviceArray d_input_dims(input_dims);
    DeviceArray d_input_strides(input_strides);
    DeviceArray d_output_dims(output_shape.dims);
    DeviceArray d_normalized_axes(normalized_axes);
    DeviceArray d_reduced_dims(reduced_dims);
    
    int threads_per_block = 256;
    int num_blocks = num_slices;
    
    // ✅ FIXED: Allocate shared memory for BOTH sum (double) and count (int64_t)
    int num_warps = (threads_per_block + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double) + num_warps * sizeof(int64_t);
    
    // Determine output C++ type
    using OutputCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    OutputCppT* output_data = output.data<OutputCppT>();
    
    cuda::reduce_mean_kernel<T, SumOpType><<<num_blocks, threads_per_block, shared_mem_size>>>(
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
    
    cudaDeviceSynchronize();
    
    return output;
}

// =================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS FOR DISPATCHER FUNCTIONS
// =================================================================

#define INSTANTIATE_VALUE_DISPATCHER(T) \
    template Tensor dispatch_reduction_gpu<T, SumOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, MinOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_reduction_gpu<T, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool);

#define INSTANTIATE_INDEX_DISPATCHER(T) \
    template Tensor dispatch_index_reduction_gpu<T, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_index_reduction_gpu<T, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_index_reduction_gpu<T, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_index_reduction_gpu<T, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool);

#define INSTANTIATE_MEAN_DISPATCHER(T) \
    template Tensor dispatch_mean_gpu<T, SumOp>(const Tensor&, const std::vector<int64_t>&, bool); \
    template Tensor dispatch_mean_gpu<T, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool);

#define INSTANTIATE_ALL_DISPATCHERS(T) \
    INSTANTIATE_VALUE_DISPATCHER(T) \
    INSTANTIATE_INDEX_DISPATCHER(T) \
    INSTANTIATE_MEAN_DISPATCHER(T)

INSTANTIATE_ALL_DISPATCHERS(int16_t)
INSTANTIATE_ALL_DISPATCHERS(int32_t)
INSTANTIATE_ALL_DISPATCHERS(int64_t)
INSTANTIATE_ALL_DISPATCHERS(float)
INSTANTIATE_ALL_DISPATCHERS(double)
INSTANTIATE_ALL_DISPATCHERS(float16_t)
INSTANTIATE_ALL_DISPATCHERS(bfloat16_t)

#endif // WITH_CUDA

} // namespace detail
} // namespace OwnTensor
/*
EXPLANATION OF TWO-LEVEL INSTANTIATION:
=======================================

FILE 1: ReductionKernels.cu
- Instantiates: __global__ void reduce_kernel<T, OpType>(...)
- These are GPU device functions
- Run ON the GPU

FILE 2: ReductionImplGPU.cu (THIS FILE)
- Instantiates: Tensor dispatch_reduction_gpu<T, OpType>(...)
- These are CPU host functions
- Run ON the CPU, but CALL the GPU kernels

WHY BOTH ARE NEEDED:
- CUDA separates compilation of device code (.cu) and host code
- Kernels compiled in ReductionKernels.cu → stored in libTensorLib.a
- Dispatchers compiled in ReductionImplGPU.cu → need to link to kernels
- Without both sets of instantiations, linker fails with "undefined reference"

WHAT GETS INSTANTIATED:
- 7 types × 14 operations = 98 dispatcher functions
- Each dispatcher calls the corresponding kernel
*/