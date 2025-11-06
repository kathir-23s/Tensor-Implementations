#if defined(WITH_CUDA)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ops/helpers/ConditionalOps.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"

namespace OwnTensor {

namespace {

// Device helpers for fp16/bf16 (reuse from ScalarOps.cu)
__device__ inline float dev_bf16_to_float(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ inline uint16_t dev_float_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

__device__ inline float dev_fp16_to_float(uint16_t bits) {
    __half h = *reinterpret_cast<__half*>(&bits);
    return __half2float(h);
}

__device__ inline uint16_t dev_float_to_fp16(float f) {
    __half h = __float2half_rn(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

// Generic where kernel
template<typename CondT, typename DataT>
__global__ void k_where(const CondT* condition, const DataT* input,
                        const DataT* other, DataT* out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        bool cond = (condition[i] != static_cast<CondT>(0));
        out[i] = cond ? input[i] : other[i];
    }
}

// Specialized kernel for fp16/bf16
template<typename CondT>
__global__ void k_where_fp16(const CondT* condition, const uint16_t* input,
                             const uint16_t* other, uint16_t* out, 
                             size_t n, int fmt) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        bool cond = (condition[i] != static_cast<CondT>(0));
        
        // Convert to float, select, convert back
        float input_val = (fmt == 1) ? dev_fp16_to_float(input[i])
                                     : dev_bf16_to_float(input[i]);
        float other_val = (fmt == 1) ? dev_fp16_to_float(other[i])
                                     : dev_bf16_to_float(other[i]);
        float result = cond ? input_val : other_val;
        
        out[i] = (fmt == 1) ? dev_float_to_fp16(result)
                            : dev_float_to_bf16(result);
    }
}

inline dim3 pick_grid(size_t n, dim3 b) {
    size_t blocks = (n + b.x - 1) / b.x;
    if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

} // anonymous namespace

void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out) {
    const size_t n = condition.numel();
    const dim3 block(256);
    const dim3 grid = pick_grid(n, block);
    
    const Dtype cond_dtype = condition.dtype();
    const Dtype input_dtype = input.dtype();
    
    dispatch_by_dtype(cond_dtype, [&](auto cond_type) {
        using CondT = decltype(cond_type);
        const CondT* cond_ptr = condition.data<CondT>();
        
        if (input_dtype == Dtype::Float16 || input_dtype == Dtype::Bfloat16) {
            int fmt = (input_dtype == Dtype::Float16) ? 1 : 2;
            const uint16_t* input_ptr = input.data<uint16_t>();
            const uint16_t* other_ptr = other.data<uint16_t>();
            uint16_t* out_ptr = out.data<uint16_t>();
            
            k_where_fp16<<<grid, block>>>(cond_ptr, input_ptr, other_ptr, 
                                          out_ptr, n, fmt);
        } else {
            dispatch_by_dtype(input_dtype, [&](auto data_type) {
                using DataT = decltype(data_type);
                const DataT* input_ptr = input.data<DataT>();
                const DataT* other_ptr = other.data<DataT>();
                DataT* out_ptr = out.data<DataT>();
                
                k_where<<<grid, block>>>(cond_ptr, input_ptr, other_ptr, 
                                         out_ptr, n);
            });
        }
    });
    
    check_cuda_error("cuda_where");
}

} // namespace OwnTensor

#endif // WITH_CUDA
