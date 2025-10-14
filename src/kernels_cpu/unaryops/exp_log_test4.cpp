#include <cmath>
#include <omp.h>
#include <stdexcept>
#include "../../../include/Tensor.h"
#include "../../../include/dispatcher/UnaryDispatcher.hpp"
#include "../../../include/dispatcher/exp_log_kernels.hpp"
#include "../../../include/dispatcher/tesnor_unaryops.hpp"
#include "../../../include/DtypeTraits.h"
#include "../../../include/types.h"

namespace exp_log {
    // outplace
    template<typename T, T(*Func)(T)> 
    void apply_unary_kernel(const T* in, T* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            out[i] = Func(in[i]);
        }
    }

    // wrappers
    float expf_wrap(float x) { return expf(x); }
    double exp_wrap(double x) { return exp(x); }
    float exp2f_wrap(float x) { return exp2f(x); }
    double exp2_wrap(double x) { return exp2(x); }
    float logf_wrap(float x) { return logf(x); }
    double log_wrap(double x) { return log(x); }
    float log2f_wrap(float x) { return log2f(x); }
    double log2_wrap(double x) { return log2(x); }
    float log10f_wrap(float x) { return log10f(x); }
    double log10_wrap(double x) { return log10(x); }
    
    // Exp
    template void apply_unary_kernel<float, expf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, exp_wrap>(const double* in, double* out, size_t size);
    // Exp2
    template void apply_unary_kernel<float, exp2f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, exp2_wrap>(const double* in, double* out, size_t size);
    // Log
    template void apply_unary_kernel<float, logf_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, log_wrap>(const double* in, double* out, size_t size);
    // Log2
    template void apply_unary_kernel<float, log2f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, log2_wrap>(const double* in, double* out, size_t size);
    // Log10
    template void apply_unary_kernel<float, log10f_wrap>(const float* in, float* out, size_t size);
    template void apply_unary_kernel<double, log10_wrap>(const double* in, double* out, size_t size);

    // inplace
    template<typename T, T(*Func)(T)>
    void inplace_dispatcher_wrapper(const T* in, T* out, size_t size) {
        apply_unary_kernel<T, Func>(in, const_cast<T*>(in), size);
    }

    // Exp In-place
    template void inplace_dispatcher_wrapper<float, expf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, exp_wrap>(const double* in, double* out, size_t size);
    // Exp2 In-place
    template void inplace_dispatcher_wrapper<float, exp2f_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, exp2_wrap>(const double* in, double* out, size_t size);
    // Log In-place
    template void inplace_dispatcher_wrapper<float, logf_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, log_wrap>(const double* in, double* out, size_t size);
    // Log2 In-place
    template void inplace_dispatcher_wrapper<float, log2f_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, log2_wrap>(const double* in, double* out, size_t size);
    // Log10 In-place
    template void inplace_dispatcher_wrapper<float, log10f_wrap>(const float* in, float* out, size_t size);
    template void inplace_dispatcher_wrapper<double, log10_wrap>(const double* in, double* out, size_t size);

    // for promotion kernel
    template<typename T_In, typename T_Out, T_Out(*T_OpFunc)(T_Out)>
    void apply_unary_promotion_kernel(const T_In* in, T_Out* out, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            T_Out temp_val = static_cast<T_Out>(in[i]);
            out[i] = T_OpFunc(temp_val);
        }
    }
    // promotion types [only Out_of_Place]
    // Exp Promotion
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::expf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::expf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::exp_wrap>(const int64_t* in, double* out, size_t size);
    // Exp2 Promotion
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::exp2f_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::exp2f_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::exp2_wrap>(const int64_t* in, double* out, size_t size);
    // Log Promotion
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::logf_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::logf_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::log_wrap>(const int64_t* in, double* out, size_t size);
    // Log2 Promotion
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::log2f_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::log2f_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::log2_wrap>(const int64_t* in, double* out, size_t size);
    // Log10 Promotion
    template void apply_unary_promotion_kernel<int16_t, float, exp_log::log10f_wrap>(const int16_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int32_t, float, exp_log::log10f_wrap>(const int32_t* in, float* out, size_t size);
    template void apply_unary_promotion_kernel<int64_t, double, exp_log::log10_wrap>(const int64_t* in, double* out, size_t size);

    // --- 16-BIT FLOAT KERNEL (Bfloat16/Float16) ---
    // This kernel converts uint16_t (F16/BF16) to float, computes the op, and converts back.
    // template<typename T_16Bit, float (*T_16Bit_To_F32)(T_16Bit), T_16Bit (*T_F32_To_16Bit)(float), float(*T_OpFunc)(float)>
    // void apply_unary_f16ish_kernel(const T_16Bit* in, T_16Bit* out, size_t size) {
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < size; ++i) {
    //         // 1. Convert 16-bit float to 32-bit float
    //         float f32_in = T_16Bit_To_F32(in[i]); 
            
    //         // 2. Compute the unary operation on 32-bit float
    //         float f32_out = T_OpFunc(f32_in);
            
    //         // 3. Convert result back to 16-bit float
    //         out[i] = T_F32_To_16Bit(f32_out); 
    //     }
    // }

    // --- Explicit Instantiations for the 16-bit float kernels ---

    // Exp: Bfloat16 -> Bfloat16
    // template void apply_unary_f16ish_kernel<
    //     uint16_t, 
    //     bfloat16_to_float,    // BF16 -> F32 conversion
    //     float_to_bfloat16,    // F32 -> BF16 conversion
    //     exp_log::expf_wrap    // The float Exp wrapper
    // >(const uint16_t* in, uint16_t* out, size_t size);

    // // Exp: Float16 -> Float16
    // template void apply_unary_f16ish_kernel<
    //     uint16_t, 
    //     float16_to_float,     // F16 -> F32 conversion
    //     float_to_float16,     // F32 -> F16 conversion
    //     exp_log::expf_wrap    // The float Exp wrapper
    // >(const uint16_t* in, uint16_t* out, size_t size);

    // ... you must repeat this for Log, Log2, etc., using their corresponding exp_log::logf_wrap, etc.

}
