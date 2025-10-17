#include <cmath>
#include "UnaryOps/exp_log.hpp"
#include "Tensor.h"

namespace OwnTensor {
// function pointers
static inline __device__ float expf_fn(float x){return expf(x);}
static inline __device__ double exp_fn(double x){return exp(x);}

// CUDA unary kernel
template<typename T, T(*func)(T)>
__global__
void unary_kernel_gpu(const T* in, T* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = func(in[idx]);
    }
}

// wrappers
Tensor exp_out_gpu_wrap(const Tensor& input_tensor){
    // assumes that the tensor already resides on GPU <====================================================================

    // fallback path for float32 or float64, etc.
    size_t size = input_tensor.numel();
    Tensor output_tensor(input_tensor.shape(), input_tensor.dtype(),
                         input_tensor.device(), input_tensor.requires_grad());

    // launch parameters
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    if (input_tensor.dtype() == Dtype::Float32) {
        const float* in_ptr = input_tensor.data<float>();
        float* out_ptr = output_tensor.data<float>();
        unary_kernel_gpu<float, expf_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else if (input_tensor.dtype() == Dtype::Float64) {
        const double* in_ptr = input_tensor.data<double>();
        double* out_ptr = output_tensor.data<double>();
        unary_kernel_gpu<double, exp_fn><<<blocks, threads>>>(in_ptr, out_ptr, size);
    } else {
        throw std::runtime_error("Unsupported dtype for exp on CUDA");
    }

    cudaDeviceSynchronize();
    return output_tensor;
}
}