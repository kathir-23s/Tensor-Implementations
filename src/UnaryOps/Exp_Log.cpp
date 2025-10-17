#include <cmath>
#include "Tensor.h"
#include "Types.h"
#include "UnaryOps/exp_log.hpp"

namespace OwnTensor {
// function pointers
static inline float expf_fn(float x){return expf(x);}
static inline double exp_fn(double x){return std::exp(x);}
// unary kernel for CPU
template<typename T_In, typename T_Out, T_Out(*Func)(T_Out)>
void unary_kernel_cpu(const T_In* in, T_Out* out, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        T_Out temp_val = static_cast<T_Out>(in[i]);
        out[i] = Func(temp_val);
    }
}
// wrappers
Tensor exp_out_cpu_wrap(const Tensor& input_tensor){
    std::cout << "Outplace is Called!\n";
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
        unary_kernel_cpu<float,float,expf_fn>(
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
    // 1. check for type promotion operations
    Dtype d = input_tensor.dtype();
    switch(d){
        case Dtype::Int16: {
            std::cout << "int16" << std::endl;
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int16_t,float,expf_fn>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            std::cout << "int32" << std::endl;
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int32_t,float,expf_fn>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            std::cout << "int64" << std::endl;
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int64_t,double,exp_fn>(
                input_tensor.data<int64_t>(),
                output_tensor.data<double>(),
                input_tensor.numel()
            );
            return output_tensor;
        }

    }
    // fallback path for float32 or float64, etc.
    Tensor output_tensor(input_tensor.shape(), input_tensor.dtype(), input_tensor.device(), input_tensor.requires_grad());
    if (input_tensor.dtype() == Dtype::Float32) {
        std::cout << "float" << std::endl;
        unary_kernel_cpu<float, float, expf_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        std::cout << "double" << std::endl;
        unary_kernel_cpu<double, double, exp_fn>(  // using std::exp
            input_tensor.data<double>(),
            output_tensor.data<double>(),
            input_tensor.numel()
        );
    }
    return output_tensor;
}
void exp_in_cpu_wrap(Tensor& input_tensor){
    // 0. check if bf16 / f16
    try{
    std::cout << "Inplace is Called!\n";
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
        //Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,expf_fn>(
            static_cast<const float*>(temp_tensor.data()),
            static_cast<float*>(temp_output.data()),
            input_tensor.numel()
        );
        // convert back to original data type
        float* temp_out_ptr = temp_output.data<float>();
        if (original_dtype == Dtype::Float16) {
        float16_t* out_ptr = input_tensor.data<float16_t>();
        for (int i = 0; i < temp_output.numel(); ++i) {
            out_ptr[i] = float16_t(temp_out_ptr[i]);
        }
        } else {
        bfloat16_t* out_ptr = input_tensor.data<bfloat16_t>();
        for (int i = 0; i < temp_output.numel(); ++i) {
            out_ptr[i] = bfloat16_t(temp_out_ptr[i]);
        }
        }
    }
    // 1. check for type promotion operations
    Dtype d = input_tensor.dtype();
    switch(d){
        case Dtype::Int16: 
        case Dtype::Int32: 
        case Dtype::Int64: throw std::runtime_error("Error: cannot do inplace for Int data types!");

    }
    // fallback path for float32 or float64, etc.
    //Tensor output_tensor(input_tensor.shape(), input_tensor.dtype(), input_tensor.device(), input_tensor.requires_grad());
    if (input_tensor.dtype() == Dtype::Float32) {
        std::cout << "float" << std::endl;
        unary_kernel_cpu<float, float, expf_fn>(
            input_tensor.data<float>(),
            input_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        std::cout << "double" << std::endl;
        unary_kernel_cpu<double, double, exp_fn>(  // using std::exp
            input_tensor.data<double>(),
            input_tensor.data<double>(),
            input_tensor.numel()
        );
    }}
    catch (const std::runtime_error& e) {
        std::cerr << "Caught runtime error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE); 
    }
}

// high level API definitions
// Out-Place
Tensor exp(const Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        std::cout << "CPU is called!" << std::endl;
        return exp_out_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        std::cout << "CUDA is called!" << std::endl;
        return exp_out_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}

Tensor exp2(const Tensor& input);
Tensor log(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor log10(const Tensor& input);

// In-Place
void exp_(Tensor& input) {
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        std::cout << "CPU is called!" << std::endl;
        exp_in_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        std::cout << "CUDA is called!" << std::endl;
        //exp_in_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
void exp2_(Tensor& input);
void log_(Tensor& input);
void log2_(Tensor& input);
void log10_(Tensor& input);

} // end of namespace OwnTensor