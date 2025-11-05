// ScalarOpsDispatch.cpp
#include <cstdint>
#include <stdexcept>
#include "ops/ScalarOps.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include <driver_types.h>
#include "device/DeviceCore.h" //✨✨✨

namespace OwnTensor {

// ---- backend declarations implemented in cpu/ScalarOps.cpp and cuda/ScalarOps.cu
void   cpu_add_inplace (Tensor&, double);
void   cpu_sub_inplace (Tensor&, double);
void   cpu_mul_inplace (Tensor&, double);
void   cpu_div_inplace (Tensor&, double);

Tensor cpu_add_copy    (const Tensor&, double);
Tensor cpu_sub_copy    (const Tensor&, double);
Tensor cpu_mul_copy    (const Tensor&, double);
Tensor cpu_div_copy    (const Tensor&, double);
Tensor cpu_sub_copy_scalar_tensor(double, const Tensor&);
Tensor cpu_div_copy_scalar_tensor(double, const Tensor&);

// CUDA backends exist only if the CUDA TU is linked; declarations are harmless here
void   cuda_add_inplace (Tensor&, double, cudaStream_t); //✨✨✨
void   cuda_sub_inplace (Tensor&, double, cudaStream_t); //✨✨✨
void   cuda_mul_inplace (Tensor&, double, cudaStream_t); //✨✨✨
void   cuda_div_inplace (Tensor&, double, cudaStream_t); //✨✨✨

Tensor cuda_add_copy    (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_sub_copy    (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_mul_copy    (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_div_copy    (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_sub_copy_scalar_tensor(double, const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_div_copy_scalar_tensor(double, const Tensor&, cudaStream_t); //✨✨✨

// ---- helpers ----
static inline bool is_integer_dtype(Dtype dt) {
    return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
}
template <typename S> static inline double to_f64(S s) { return static_cast<double>(s); }

// ======================= Public API =======================
template<typename S>
Tensor& operator+=(Tensor& t, S s) {
    const double sd = to_f64(s);
    if (sd == 0.0) return t;
    if (t.device().is_cuda()) 
    {  //✨✨✨
        #ifdef WITH_CUDA //✨✨✨
        cuda_add_inplace(t, sd, OwnTensor::cuda::getCurrentStream()); // <-- Use context //✨✨✨
        #endif //✨✨✨
    }  else  {    cpu_add_inplace(t, sd);} //✨✨✨
    return t;
}

template<typename S>
Tensor& operator-=(Tensor& t, S s) {
    const double sd = to_f64(s);
    if (sd == 0.0) return t;
    if (t.device().is_cuda())
    { //✨✨✨
        #ifdef WITH_CUDA
        cuda_sub_inplace(t, sd, OwnTensor::cuda::getCurrentStream()); // <-- Use context
        #endif
    }  else  {cpu_sub_inplace(t, sd);}
    return t; //✨✨✨
}

template<typename S>
Tensor& operator*=(Tensor& t, S s) {
    const double sd = to_f64(s);
    if (sd == 1.0) return t;
    if (t.device().is_cuda()) { //✨✨✨
        #ifdef WITH_CUDA
        cuda_mul_inplace(t, sd, OwnTensor::cuda::getCurrentStream());
        #endif    
    }     else    {cpu_mul_inplace(t, sd);}
    return t; //✨✨✨
}

template<typename S>
Tensor& operator/=(Tensor& t, S s) {
    const double sd = to_f64(s);
    if (sd == 1.0) return t;
    if (!t.device().is_cuda() && is_integer_dtype(t.dtype()) && sd == 0.0)
        throw std::runtime_error("Division by zero");
    if (t.device().is_cuda())  { //✨✨✨
        #ifdef WITH_CUDA
        cuda_div_inplace(t, sd, OwnTensor::cuda::getCurrentStream());
    #endif
    }  else  {  cpu_div_inplace(t, sd);}
    return t; //✨✨✨
}

template<typename S>
Tensor operator+(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_add_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_add_copy(a, to_f64(s)); //✨✨✨
}
template<typename S>
Tensor operator-(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_sub_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_sub_copy(a, to_f64(s)); //✨✨✨
}
template<typename S>
Tensor operator*(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_mul_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_mul_copy(a, to_f64(s)); //✨✨✨
}
template<typename S>
Tensor operator/(const Tensor& a, S s) {
    const double sd = to_f64(s);
    if (!a.device().is_cuda() && is_integer_dtype(a.dtype()) && sd == 0.0)
        throw std::runtime_error("Division by zero");
    return a.device().is_cuda() ? cuda_div_copy(a, sd, OwnTensor::cuda::getCurrentStream()) : cpu_div_copy(a, sd); //✨✨✨
}

template<typename S>
Tensor operator+(S s, const Tensor& a) { return a + s; }

template<typename S>
Tensor operator-(S s, const Tensor& a) {
    return a.device().is_cuda() ? cuda_sub_copy_scalar_tensor(to_f64(s), a, OwnTensor::cuda::getCurrentStream()) //✨✨✨
                                : cpu_sub_copy_scalar_tensor(to_f64(s), a);
}

template<typename S>
Tensor operator*(S s, const Tensor& a) { return a * s; }

template<typename S>
Tensor operator/(S s, const Tensor& a) {
    return a.device().is_cuda() ? cuda_div_copy_scalar_tensor(to_f64(s), a, OwnTensor::cuda::getCurrentStream()) //✨✨✨
                                : cpu_div_copy_scalar_tensor(to_f64(s), a);
}

// ======================= Explicit instantiations =======================
using OwnTensor::float16_t;
using OwnTensor::bfloat16_t;

template Tensor& operator+=<int16_t>(Tensor&, int16_t);
template Tensor& operator+=<int32_t>(Tensor&, int32_t);
template Tensor& operator+=<int64_t>(Tensor&, int64_t);
template Tensor& operator+=<float>(Tensor&, float);
template Tensor& operator+=<double>(Tensor&, double);
template Tensor& operator+=<float16_t>(Tensor&, float16_t);
template Tensor& operator+=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator-=<int16_t>(Tensor&, int16_t);
template Tensor& operator-=<int32_t>(Tensor&, int32_t);
template Tensor& operator-=<int64_t>(Tensor&, int64_t);
template Tensor& operator-=<float>(Tensor&, float);
template Tensor& operator-=<double>(Tensor&, double);
template Tensor& operator-=<float16_t>(Tensor&, float16_t);
template Tensor& operator-=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator*=<int16_t>(Tensor&, int16_t);
template Tensor& operator*=<int32_t>(Tensor&, int32_t);
template Tensor& operator*=<int64_t>(Tensor&, int64_t);
template Tensor& operator*=<float>(Tensor&, float);
template Tensor& operator*=<double>(Tensor&, double);
template Tensor& operator*=<float16_t>(Tensor&, float16_t);
template Tensor& operator*=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor& operator/=<int16_t>(Tensor&, int16_t);
template Tensor& operator/=<int32_t>(Tensor&, int32_t);
template Tensor& operator/=<int64_t>(Tensor&, int64_t);
template Tensor& operator/=<float>(Tensor&, float);
template Tensor& operator/=<double>(Tensor&, double);
template Tensor& operator/=<float16_t>(Tensor&, float16_t);
template Tensor& operator/=<bfloat16_t>(Tensor&, bfloat16_t);

template Tensor operator+<int16_t>(const Tensor&, int16_t);
template Tensor operator+<int32_t>(const Tensor&, int32_t);
template Tensor operator+<int64_t>(const Tensor&, int64_t);
template Tensor operator+<float>(const Tensor&, float);
template Tensor operator+<double>(const Tensor&, double);
template Tensor operator+<float16_t>(const Tensor&, float16_t);
template Tensor operator+<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator-<int16_t>(const Tensor&, int16_t);
template Tensor operator-<int32_t>(const Tensor&, int32_t);
template Tensor operator-<int64_t>(const Tensor&, int64_t);
template Tensor operator-<float>(const Tensor&, float);
template Tensor operator-<double>(const Tensor&, double);
template Tensor operator-<float16_t>(const Tensor&, float16_t);
template Tensor operator-<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator*<int16_t>(const Tensor&, int16_t);
template Tensor operator*<int32_t>(const Tensor&, int32_t);
template Tensor operator*<int64_t>(const Tensor&, int64_t);
template Tensor operator*<float>(const Tensor&, float);
template Tensor operator*<double>(const Tensor&, double);
template Tensor operator*<float16_t>(const Tensor&, float16_t);
template Tensor operator*<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator/<int16_t>(const Tensor&, int16_t);
template Tensor operator/<int32_t>(const Tensor&, int32_t);
template Tensor operator/<int64_t>(const Tensor&, int64_t);
template Tensor operator/<float>(const Tensor&, float);
template Tensor operator/<double>(const Tensor&, double);
template Tensor operator/<float16_t>(const Tensor&, float16_t);
template Tensor operator/<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator+<int16_t>(int16_t, const Tensor&);
template Tensor operator+<int32_t>(int32_t, const Tensor&);
template Tensor operator+<int64_t>(int64_t, const Tensor&);
template Tensor operator+<float>(float, const Tensor&);
template Tensor operator+<double>(double, const Tensor&);
template Tensor operator+<float16_t>(float16_t, const Tensor&);
template Tensor operator+<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator-<int16_t>(int16_t, const Tensor&);
template Tensor operator-<int32_t>(int32_t, const Tensor&);
template Tensor operator-<int64_t>(int64_t, const Tensor&);
template Tensor operator-<float>(float, const Tensor&);
template Tensor operator-<double>(double, const Tensor&);
template Tensor operator-<float16_t>(float16_t, const Tensor&);
template Tensor operator-<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator*<int16_t>(int16_t, const Tensor&);
template Tensor operator*<int32_t>(int32_t, const Tensor&);
template Tensor operator*<int64_t>(int64_t, const Tensor&);
template Tensor operator*<float>(float, const Tensor&);
template Tensor operator*<double>(double, const Tensor&);
template Tensor operator*<float16_t>(float16_t, const Tensor&);
template Tensor operator*<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator/<int16_t>(int16_t, const Tensor&);
template Tensor operator/<int32_t>(int32_t, const Tensor&);
template Tensor operator/<int64_t>(int64_t, const Tensor&);
template Tensor operator/<float>(float, const Tensor&);
template Tensor operator/<double>(double, const Tensor&);
template Tensor operator/<float16_t>(float16_t, const Tensor&);
template Tensor operator/<bfloat16_t>(bfloat16_t, const Tensor&);

} // namespace OwnTensor
