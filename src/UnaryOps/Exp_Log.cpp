#include <cmath>
#include "../../include/Tensor.h"
#include "../../include/Types.h"
#include "../../include/UnaryOps/exp_log.hpp"

namespace OwnTensor {
// function pointers
static inline float expf_fn(float x){return expf(x);}
static inline double exp_fn(double x){return std::exp(x);}

// unary kernel definition
template<typename T, T(*func)(T)>
void unary_kernel(const T* in, T* out, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        out[i] = func(in[i]);
    }
}

// wrappers
Tensor exp_cpu_wrap(const Tensor& input_tensor){
    // 0. check if bf16 / f16
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        std::cout << "f16s caught!" << std::endl;
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            std::cout << "f16 caught!" << std::endl;
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            std::cout << "bf16 caught!" << std::endl;
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel<float, expf_fn>(
            static_cast<const float*>(temp_tensor.data()),
            static_cast<float*>(temp_output.data()),
            input_tensor.numel()
        );

        // convert back to original data type
        float* temp_out_ptr = temp_output.data<float>();
        if (original_dtype == Dtype::Float16) {
        float16_t* out_ptr = output_tensor.data<float16_t>();
        for (int i = 0; i < temp_output.numel(); ++i) {
            out_ptr[i] = float16_t(temp_out_ptr[i]);
        }
        } else {
        bfloat16_t* out_ptr = output_tensor.data<bfloat16_t>();
        for (int i = 0; i < temp_output.numel(); ++i) {
            out_ptr[i] = bfloat16_t(temp_out_ptr[i]);
        }
        }
        return output_tensor;
    }

    // fallback path for float32 or float64, etc.
    Tensor output_tensor(input_tensor.shape(), input_tensor.dtype(), input_tensor.device(), input_tensor.requires_grad());

    if (input_tensor.dtype() == Dtype::Float32) {
        unary_kernel<float, expf_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel<double, exp_fn>(  // using std::exp
            input_tensor.data<double>(),
            output_tensor.data<double>(),
            input_tensor.numel()
        );
    }

    return output_tensor;
    // 1. check for type promotion operations

    // 2. get the metadata from the input tensor

    // 3. create the output tensor

    // 4. call the kernel
}


// high level API definitions
// Out-Place
Tensor exp(const Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        std::cout << "CPU is called!" << std::endl;
        std::cin.get();
        return exp_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        std::cout << "CUDA is called!" << std::endl;
        return exp_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
Tensor exp2(const Tensor& input);
Tensor log(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor log10(const Tensor& input);
// In-Place
void exp_(Tensor& input);
void exp2_(Tensor& input);
void log_(Tensor& input);
void log2_(Tensor& input);
void log10_(Tensor& input);

} // end of namespace OwnTensor