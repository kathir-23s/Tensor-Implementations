#include <cmath>
#include "Tensor.h"
#include "Types.h"
#include "UnaryOps/exp_log.hpp"

namespace OwnTensor {
// function pointers
static inline float expf_fn(float x){return expf(x);}
static inline double exp_fn(double x){return std::exp(x);}
static inline float exp2f_fn(float x){return exp2f(x);}
static inline double exp2_fn(double x){return std::exp2(x);}
static inline float logf_fn(float x){return logf(x);}
static inline double log_fn(double x){return std::log(x);}
static inline float log2f_fn(float x){return log2f(x);}
static inline double log2_fn(double x){return std::log2(x);}
static inline float log10f_fn(float x){return log10f(x);}
static inline double log10_fn(double x){return std::log10(x);}
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
    // 0. check if bf16 / f16
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
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
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int16_t,float,expf_fn>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int32_t,float,expf_fn>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
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
        unary_kernel_cpu<float, float, expf_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    } else if (input_tensor.dtype() == Dtype::Float64) {
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
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
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
        unary_kernel_cpu<float, float, expf_fn>(
            input_tensor.data<float>(),
            input_tensor.data<float>(),
            input_tensor.numel()
        );
    } else if (input_tensor.dtype() == Dtype::Float64) {
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

Tensor exp2_out_cpu_wrap(const Tensor& input_tensor){
    // 0. check if bf16 / f16
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,exp2f_fn>(
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
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int16_t,float,exp2f_fn>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int32_t,float,exp2f_fn>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int64_t,double,exp2_fn>(
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
        unary_kernel_cpu<float, float, exp2f_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    } else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, exp_fn>(  // using std::exp
            input_tensor.data<double>(),
            output_tensor.data<double>(),
            input_tensor.numel()
        );
    }
    return output_tensor;
}
void exp2_in_cpu_wrap(Tensor& input_tensor){
    // 0. check if bf16 / f16
    try{
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        //Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,exp2f_fn>(
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
        unary_kernel_cpu<float, float, exp2f_fn>(
            input_tensor.data<float>(),
            input_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, exp2_fn>(  // using std::exp
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

Tensor log_out_cpu_wrap(const Tensor& input_tensor){
    // 0. check if bf16 / f16
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,logf_fn>(
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
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int16_t,float,logf_fn>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int32_t,float,logf_fn>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int64_t,double,log_fn>(
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
        unary_kernel_cpu<float, float, logf_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, log_fn>(  // using std::exp
            input_tensor.data<double>(),
            output_tensor.data<double>(),
            input_tensor.numel()
        );
    }
    return output_tensor;
}
void log_in_cpu_wrap(Tensor& input_tensor){
    // 0. check if bf16 / f16
    try{
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        //Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,logf_fn>(
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
        unary_kernel_cpu<float, float, logf_fn>(
            input_tensor.data<float>(),
            input_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, log_fn>(  // using std::exp
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

Tensor log2_out_cpu_wrap(const Tensor& input_tensor){
    // 0. check if bf16 / f16
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,log2f_fn>(
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
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int16_t,float,log2f_fn>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int32_t,float,log2f_fn>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int64_t,double,log2_fn>(
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
        unary_kernel_cpu<float, float, log2f_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, log2_fn>(  // using std::exp
            input_tensor.data<double>(),
            output_tensor.data<double>(),
            input_tensor.numel()
        );
    }
    return output_tensor;
}
void log2_in_cpu_wrap(Tensor& input_tensor){
    // 0. check if bf16 / f16
    try{
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        //Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,log2f_fn>(
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
        unary_kernel_cpu<float, float, log2f_fn>(
            input_tensor.data<float>(),
            input_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, log2_fn>(  // using std::exp
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

Tensor log10_out_cpu_wrap(const Tensor& input_tensor){
    // 0. check if bf16 / f16
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,log10f_fn>(
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
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int16_t,float,log10f_fn>(
                input_tensor.data<int16_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int32: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int32_t,float,log10f_fn>(
                input_tensor.data<int32_t>(),
                output_tensor.data<float>(),
                input_tensor.numel()
            );
            return output_tensor;
        }
        case Dtype::Int64: {
            Tensor output_tensor(input_tensor.shape(), Dtype::Float64, input_tensor.device(), input_tensor.requires_grad());
            unary_kernel_cpu<int64_t,double,log10_fn>(
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
        unary_kernel_cpu<float, float, log10f_fn>(
            input_tensor.data<float>(),
            output_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, log10_fn>(  // using std::exp
            input_tensor.data<double>(),
            output_tensor.data<double>(),
            input_tensor.numel()
        );
    }
    return output_tensor;
}
void log10_in_cpu_wrap(Tensor& input_tensor){
    // 0. check if bf16 / f16
    try{
    if ((input_tensor.dtype() == Dtype::Bfloat16) || (input_tensor.dtype() == Dtype::Float16)) {
        // store original dtype to restore later
        Dtype original_dtype = input_tensor.dtype();
        // create a temporary f32 tensor for computation
        Tensor temp_tensor(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // convert input (f16/bf16) -> f32
        float* temp_ptr = temp_tensor.data<float>();
        if (original_dtype == Dtype::Float16) {
            const float16_t* in_ptr = input_tensor.data<float16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }
        } else {
            const bfloat16_t* in_ptr = input_tensor.data<bfloat16_t>();
            for (int i = 0; i < input_tensor.numel(); ++i) {
                temp_ptr[i] = static_cast<float>(in_ptr[i]);
            }   
        }
        // perform the operation
        //Tensor output_tensor(input_tensor.shape(), original_dtype, input_tensor.device(), input_tensor.requires_grad());
        Tensor temp_output(input_tensor.shape(), Dtype::Float32, input_tensor.device(), input_tensor.requires_grad());
        // call the kernel using float32 data pointers
        unary_kernel_cpu<float,float,log10f_fn>(
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
        unary_kernel_cpu<float, float, log10f_fn>(
            input_tensor.data<float>(),
            input_tensor.data<float>(),
            input_tensor.numel()
        );
    }
    else if (input_tensor.dtype() == Dtype::Float64) {
        unary_kernel_cpu<double, double, log10_fn>(  // using std::exp
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
        return exp_out_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        return exp_out_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
Tensor exp2(const Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return exp2_out_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        return exp2_out_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
Tensor log(const Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return log_out_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        return log_out_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
Tensor log2(const Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return log2_out_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        return log2_out_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
Tensor log10(const Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return log10_out_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        return log10_out_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}

// In-Place
void exp_(Tensor& input) {
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        exp_in_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        exp_in_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
void exp2_(Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        exp2_in_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        exp2_in_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
void log_(Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        log_in_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        log_in_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
void log2_(Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        log2_in_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        log2_in_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}
void log10_(Tensor& input){
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        log10_in_cpu_wrap(input);       // CPU implementation
    } else if (dev.is_cuda()) {
        // make sure the tensor resides on the GPU
        log10_in_gpu_wrap(input);   // GPU implementation
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}

} // end of namespace OwnTensor