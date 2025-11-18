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

Tensor cpu_eq_copy    (const Tensor&, double);
Tensor cpu_neq_copy   (const Tensor&, double);
Tensor cpu_leq_copy (const Tensor&, double);
Tensor cpu_geq_copy (const Tensor&, double);    
Tensor cpu_lt_copy (const Tensor&, double);
Tensor cpu_gt_copy (const Tensor&, double);
Tensor cpu_s_leq_copy( double,const Tensor&);
Tensor cpu_s_geq_copy( double,const Tensor&);
Tensor cpu_s_lt_copy ( double,const Tensor&);
Tensor cpu_s_gt_copy ( double,const Tensor&);

Tensor cpu_eq_copy    (const Tensor&, double);
Tensor cpu_neq_copy   (const Tensor&, double);
Tensor cpu_leq_copy (const Tensor&, double);
Tensor cpu_geq_copy (const Tensor&, double);    
Tensor cpu_lt_copy (const Tensor&, double);
Tensor cpu_gt_copy (const Tensor&, double);
Tensor cpu_s_leq_copy( double,const Tensor&);
Tensor cpu_s_geq_copy( double,const Tensor&);
Tensor cpu_s_lt_copy ( double,const Tensor&);
Tensor cpu_s_gt_copy ( double,const Tensor&);

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

Tensor cuda_eq_copy    (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_neq_copy   (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_leq_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_geq_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_lt_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_gt_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_s_leq_copy( double,const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_s_geq_copy( double,const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_s_lt_copy ( double,const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_s_gt_copy ( double,const Tensor&, cudaStream_t); //✨✨✨

Tensor cuda_eq_copy    (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_neq_copy   (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_leq_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_geq_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_lt_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_gt_copy (const Tensor&, double, cudaStream_t); //✨✨✨
Tensor cuda_s_leq_copy( double,const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_s_geq_copy( double,const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_s_lt_copy ( double,const Tensor&, cudaStream_t); //✨✨✨
Tensor cuda_s_gt_copy ( double,const Tensor&, cudaStream_t); //✨✨✨

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
    // std::cout<<"hi"<<std::endl;
    // std::cout<<"hi"<<std::endl;
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

template<typename S>
Tensor operator==(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_eq_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_eq_copy(a, to_f64(s)); //✨✨✨
}

template<typename S>
Tensor operator!=(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_neq_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_neq_copy(a, to_f64(s)); //✨✨✨
}

template<typename S>
Tensor operator<=(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_leq_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_leq_copy(a, to_f64(s)); //✨✨✨
}
template<typename S>
Tensor operator>=(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_geq_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_geq_copy(a, to_f64(s)); //✨✨✨
}
template<typename S>
Tensor operator>(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_gt_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_gt_copy(a, to_f64(s)); //✨✨✨
}

template<typename S>
Tensor operator<(const Tensor& a, S s) {
    return a.device().is_cuda() ? cuda_lt_copy(a, to_f64(s), OwnTensor::cuda::getCurrentStream()) : cpu_lt_copy(a, to_f64(s)); //✨✨✨
}
template<typename S>
Tensor operator==( S s,const Tensor& a) {
    return (a == s);
}
template<typename S>
Tensor operator!=( S s,const Tensor& a) {
    return (a != s);}

template<typename S>
Tensor operator<=( S s,const Tensor& a) {
    return a.device().is_cuda() ? cuda_s_leq_copy(to_f64(s),a, OwnTensor::cuda::getCurrentStream()) : cpu_s_leq_copy(to_f64(s),a); //✨✨✨
}
template<typename S>
Tensor operator>=( S s,const Tensor& a) {
    return a.device().is_cuda() ? cuda_s_geq_copy(to_f64(s),a, OwnTensor::cuda::getCurrentStream()) : cpu_s_geq_copy(to_f64(s),a); //✨✨✨
}
template<typename S>
Tensor operator>( S s,const Tensor& a) {
    return a.device().is_cuda() ? cuda_s_gt_copy(to_f64(s),a, OwnTensor::cuda::getCurrentStream()) : cpu_s_gt_copy(to_f64(s),a); //✨✨✨
}

template<typename S>
Tensor operator<(S s,const Tensor& a) {
    return a.device().is_cuda() ? cuda_s_lt_copy(to_f64(s),a, OwnTensor::cuda::getCurrentStream()) : cpu_s_lt_copy(to_f64(s),a ); //✨✨✨
}

template<typename S>
Tensor logical_AND([[maybe_unused]]const Tensor& a, [[maybe_unused]]S b) {
    throw std::runtime_error("logical_AND with scalar is not supported. Try Tensor logical_AND( Tensor, Tensor)");
}

template<typename S>
Tensor logical_OR([[maybe_unused]]const Tensor& a,[[maybe_unused]] S b) {
    throw std::runtime_error("logical_OR with scalar is not supported. Try Tensor logical_OR( Tensor, Tensor)");
}

template<typename S>
Tensor logical_XOR([[maybe_unused]]const Tensor& a,[[maybe_unused]] S b) {
    throw std::runtime_error("logical_XOR with scalar is not supported. Try Tensor logical_XOR( Tensor, Tensor)");
}

template<typename S>
Tensor logical_NOT([[maybe_unused]]S a) {
    throw std::runtime_error("logical_NOT with scalar is not supported. Try Tensor logical_NOT( Tensor )");
}

template<typename S>
Tensor logical_AND([[maybe_unused]]S a, [[maybe_unused]]const Tensor& b) {
    throw std::runtime_error("logical_AND with scalar is not supported. Try Tensor logical_AND( Tensor, Tensor)");
}

template<typename S>
Tensor logical_OR([[maybe_unused]]S a,[[maybe_unused]] const Tensor& b) {
    throw std::runtime_error("logical_OR with scalar is not supported. Try Tensor logical_OR( Tensor, Tensor)");
}

template<typename S>
Tensor logical_XOR([[maybe_unused]]S a, [[maybe_unused]]const Tensor& b) {
    throw std::runtime_error("logical_XOR with scalar is not supported. Try Tensor logical_XOR( Tensor, Tensor)");
}

//======================= Explicit instantiations =======================
// using OwnTensor::float16_t;
// using OwnTensor::bfloat16_t;

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

template Tensor operator==<int16_t>(int16_t, const Tensor&);
template Tensor operator==<int32_t>(int32_t, const Tensor&);
template Tensor operator==<int64_t>(int64_t, const Tensor&);
template Tensor operator==<float>(float, const Tensor&);
template Tensor operator==<double>(double, const Tensor&);
template Tensor operator==<float16_t>(float16_t, const Tensor&);
template Tensor operator==<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator!=<int16_t>(int16_t, const Tensor&);
template Tensor operator!=<int32_t>(int32_t, const Tensor&);
template Tensor operator!=<int64_t>(int64_t, const Tensor&);
template Tensor operator!=<float>(float, const Tensor&);
template Tensor operator!=<double>(double, const Tensor&);
template Tensor operator!=<float16_t>(float16_t, const Tensor&);
template Tensor operator!=<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator>=<int16_t>(int16_t, const Tensor&);
template Tensor operator>=<int32_t>(int32_t, const Tensor&);
template Tensor operator>=<int64_t>(int64_t, const Tensor&);
template Tensor operator>=<float>(float, const Tensor&);
template Tensor operator>=<double>(double, const Tensor&);
template Tensor operator>=<float16_t>(float16_t, const Tensor&);
template Tensor operator>=<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator<=<int16_t>(int16_t, const Tensor&);
template Tensor operator<=<int32_t>(int32_t, const Tensor&);
template Tensor operator<=<int64_t>(int64_t, const Tensor&);
template Tensor operator<=<float>(float, const Tensor&);
template Tensor operator<=<double>(double, const Tensor&);
template Tensor operator<=<float16_t>(float16_t, const Tensor&);
template Tensor operator<=<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator><int16_t>(int16_t, const Tensor&);
template Tensor operator><int32_t>(int32_t, const Tensor&);
template Tensor operator><int64_t>(int64_t, const Tensor&);
template Tensor operator><float>(float, const Tensor&);
template Tensor operator><double>(double, const Tensor&);
template Tensor operator><float16_t>(float16_t, const Tensor&);
template Tensor operator><bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator< <int16_t>(int16_t, const Tensor&);
template Tensor operator< <int32_t>(int32_t, const Tensor&);
template Tensor operator< <int64_t>(int64_t, const Tensor&);
template Tensor operator< <float>(float, const Tensor&);
template Tensor operator< <double>(double, const Tensor&);
template Tensor operator< <float16_t>(float16_t, const Tensor&);
template Tensor operator< <bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor operator==<int16_t>(const Tensor&, int16_t);
template Tensor operator==<int32_t>(const Tensor&, int32_t);
template Tensor operator==<int64_t>(const Tensor&, int64_t);
template Tensor operator==<float>(const Tensor&, float);
template Tensor operator==<double>(const Tensor&, double);
template Tensor operator==<float16_t>(const Tensor&, float16_t);
template Tensor operator==<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator!=<int16_t>(const Tensor&, int16_t);
template Tensor operator!=<int32_t>(const Tensor&, int32_t);
template Tensor operator!=<int64_t>(const Tensor&, int64_t);
template Tensor operator!=<float>(const Tensor&, float);
template Tensor operator!=<double>(const Tensor&, double);
template Tensor operator!=<float16_t>(const Tensor&, float16_t);
template Tensor operator!=<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator>=<int16_t>(const Tensor&, int16_t);
template Tensor operator>=<int32_t>(const Tensor&, int32_t);
template Tensor operator>=<int64_t>(const Tensor&, int64_t);
template Tensor operator>=<float>(const Tensor&, float);
template Tensor operator>=<double>(const Tensor&, double);
template Tensor operator>=<float16_t>(const Tensor&, float16_t);
template Tensor operator>=<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator<=<int16_t>(const Tensor&, int16_t);
template Tensor operator<=<int32_t>(const Tensor&, int32_t);
template Tensor operator<=<int64_t>(const Tensor&, int64_t);
template Tensor operator<=<float>(const Tensor&, float);
template Tensor operator<=<double>(const Tensor&, double);
template Tensor operator<=<float16_t>(const Tensor&, float16_t);
template Tensor operator<=<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator><int16_t>(const Tensor&, int16_t);
template Tensor operator><int32_t>(const Tensor&, int32_t);
template Tensor operator><int64_t>(const Tensor&, int64_t);
template Tensor operator><float>(const Tensor&, float);
template Tensor operator><double>(const Tensor&, double);
template Tensor operator><float16_t>(const Tensor&, float16_t);
template Tensor operator><bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor operator< <int16_t>(const Tensor&, int16_t);
template Tensor operator< <int32_t>(const Tensor&, int32_t);
template Tensor operator< <int64_t>(const Tensor&, int64_t);
template Tensor operator< <float>(const Tensor&, float);
template Tensor operator< <double>(const Tensor&, double);
template Tensor operator< <float16_t>(const Tensor&, float16_t);
template Tensor operator< <bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor logical_AND<int16_t>(const Tensor&, int16_t);
template Tensor logical_AND<int32_t>(const Tensor&, int32_t);
template Tensor logical_AND<int64_t>(const Tensor&, int64_t);
template Tensor logical_AND<float>(const Tensor&, float);
template Tensor logical_AND<double>(const Tensor&, double);
template Tensor logical_AND<float16_t>(const Tensor&, float16_t);
template Tensor logical_AND<bfloat16_t>(const Tensor&, bfloat16_t);

template Tensor logical_OR<int16_t>(const Tensor&, int16_t);
template Tensor logical_OR<int32_t>(const Tensor&, int32_t);
template Tensor logical_OR<int64_t>(const Tensor&, int64_t);
template Tensor logical_OR<float>(const Tensor&, float);
template Tensor logical_OR<double>(const Tensor&, double);
template Tensor logical_OR<float16_t>(const Tensor&, float16_t);
template Tensor logical_OR<bfloat16_t>(const Tensor&, bfloat16_t);  

template Tensor logical_XOR<int16_t>(const Tensor&, int16_t);
template Tensor logical_XOR<int32_t>(const Tensor&, int32_t);
template Tensor logical_XOR<int64_t>(const Tensor&, int64_t);
template Tensor logical_XOR<float>(const Tensor&, float);
template Tensor logical_XOR<double>(const Tensor&, double);
template Tensor logical_XOR<float16_t>(const Tensor&, float16_t);
template Tensor logical_XOR<bfloat16_t>(const Tensor&, bfloat16_t); 

template Tensor logical_NOT<int16_t>(int16_t);
template Tensor logical_NOT<int32_t>(int32_t);
template Tensor logical_NOT<int64_t>(int64_t);
template Tensor logical_NOT<float>(float);
template Tensor logical_NOT<double>(double);
template Tensor logical_NOT<float16_t>(float16_t);
template Tensor logical_NOT<bfloat16_t>(bfloat16_t);

template Tensor logical_AND<int16_t>(int16_t, const Tensor&);
template Tensor logical_AND<int32_t>(int32_t, const Tensor&);
template Tensor logical_AND<int64_t>(int64_t, const Tensor&);
template Tensor logical_AND<float>(float, const Tensor&);
template Tensor logical_AND<double>(double, const Tensor&);
template Tensor logical_AND<float16_t>(float16_t, const Tensor&);
template Tensor logical_AND<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor logical_OR<int16_t>(int16_t, const Tensor&);
template Tensor logical_OR<int32_t>(int32_t, const Tensor&);
template Tensor logical_OR<int64_t>(int64_t, const Tensor&);
template Tensor logical_OR<float>(float, const Tensor&);
template Tensor logical_OR<double>(double, const Tensor&);
template Tensor logical_OR<float16_t>(float16_t, const Tensor&);
template Tensor logical_OR<bfloat16_t>(bfloat16_t, const Tensor&);

template Tensor logical_XOR<int16_t>(int16_t, const Tensor&);
template Tensor logical_XOR<int32_t>(int32_t, const Tensor&);
template Tensor logical_XOR<int64_t>(int64_t, const Tensor&);
template Tensor logical_XOR<float>(float, const Tensor&);
template Tensor logical_XOR<double>(double, const Tensor&);
template Tensor logical_XOR<float16_t>(float16_t, const Tensor&);
template Tensor logical_XOR<bfloat16_t>(bfloat16_t, const Tensor&);

} // namespace OwnTensor
