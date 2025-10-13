// UnaryUnifiedDispatcher.cpp
// Central dispatcher + registration. All kernels live in the 3 cpp files.

// #include "../include/UnaryUnifiedDispatcher.h"
#include "UnaryUnifiedDispatcher.h"
#include "Types.h"

#include <unordered_map>
#include <mutex>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <type_traits>

using namespace OwnTensor::detail;

namespace OwnTensor {
// ===============================
// Helpers
// ===============================

static inline bool is_f16(Dtype d){
  return d==Dtype::Float16 || d==Dtype::Bfloat16;
}
static inline bool op_needs_param(UnaryOp op){ return op==UnaryOp::Power; }
static inline bool op_promotes_ints(UnaryOp op) {
  switch (op) {
    case UnaryOp::Sin: case UnaryOp::Cos: case UnaryOp::Tan:
    case UnaryOp::Asin: case UnaryOp::Acos: case UnaryOp::Atan:
    case UnaryOp::Sinh: case UnaryOp::Cosh: case UnaryOp::Tanh:
    case UnaryOp::Asinh: case UnaryOp::Acosh: case UnaryOp::Atanh:
    case UnaryOp::Exp: case UnaryOp::Exp2:
    case UnaryOp::Log: case UnaryOp::Log2: case UnaryOp::Log10:
    case UnaryOp::Sqrt:                    
    case UnaryOp::Reciprocal:  
      return true; // compute in float for integer inputs
    default: return false;
  }
}
static inline Dtype promoted_dtype_for(const Tensor& x, UnaryOp op) {
  if (op_promotes_ints(op) && is_int(x.dtype())) return Dtype::Float32;
  return x.dtype();
}

static inline void f16_to_f32(const uint16_t* in, float* out, size_t n, bool bf16){
  if (!bf16) { for(size_t i=0;i<n;++i) out[i]=float16_to_float(in[i]); }
  else       { for(size_t i=0;i<n;++i) out[i]=bfloat16_to_float(in[i]); }
}
static inline void f32_to_f16(const float* in, uint16_t* out, size_t n, bool bf16){
  if (!bf16) { for(size_t i=0;i<n;++i) out[i]=float_to_float16(in[i]); }
  else       { for(size_t i=0;i<n;++i) out[i]=float_to_bfloat16(in[i]); }
}

// ===============================
// Type-erased kernel signatures
// ===============================
using OutFn      = void(*)(const void* in, void* out, size_t n);
using InFn       = void(*)(void* data, size_t n);
using OutParamFn = void(*)(const void* in, void* out, size_t n, double p);
using InParamFn  = void(*)(void* data, size_t n, double p);

// ===============================
// External kernels (provided by 3 files)
// ===============================

// --- Trigonometry.cpp ---
namespace trig_detail {
  enum class UnaryKind {
    Sin, Cos, Tan, Asin, Acos, Atan,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
  };
  template<typename T, UnaryKind K>
  void apply_unary_kernel(const T* in, T* out, size_t n);
}

// --- ExpLogKernels.cpp ---
namespace exp_log_detail {
  void exp_float_kernel   (const void*, void*, size_t);
  void exp_double_kernel  (const void*, void*, size_t);
  void exp2_float_kernel  (const void*, void*, size_t);
  void exp2_double_kernel (const void*, void*, size_t);
  void log_float_kernel   (const void*, void*, size_t);
  void log_double_kernel  (const void*, void*, size_t);
  void log2_float_kernel  (const void*, void*, size_t);
  void log2_double_kernel (const void*, void*, size_t);
  void log10_float_kernel (const void*, void*, size_t);
  void log10_double_kernel(const void*, void*, size_t);
}

// --- Algebraic.cpp ---
namespace alg_detail {
  template<typename T> void square_impl(T*, size_t);
  template<typename T> void sqrt_impl(T*, size_t);
  template<typename T> void negate_impl(T*, size_t);
  template<typename T> void abs_impl(T*, size_t);
  template<typename T> void sign_impl(T*, size_t);
  template<typename T> void reciprocal_impl(T*, size_t);

  template<typename T> void square2_impl(const T*, T*, size_t);
  template<typename T> void sqrt2_impl  (const T*, T*, size_t);
  template<typename T> void negate2_impl(const T*, T*, size_t);
  template<typename T> void abs2_impl   (const T*, T*, size_t);
  template<typename T> void sign2_impl  (const T*, T*, size_t);
  template<typename T> void reciprocal2_impl(const T*, T*, size_t);

  template<typename T> void power_impl (T*, size_t, double);
  template<typename T> void power2_impl(const T*, T*, size_t, double);
}

// ===============================
// Registry
// ===============================
struct OpKey {
  UnaryOp op;
  Dtype dt;
  bool operator==(const OpKey& o) const noexcept { return op==o.op && dt==o.dt; }
};
struct OpKeyHash {
  size_t operator()(const OpKey& k) const noexcept {
    const size_t a = static_cast<size_t>(k.op);
    const size_t b = static_cast<size_t>(k.dt);
    return (a * 1315423911u) ^ (b + 0x9e3779b97f4a7c15ULL + (a<<6) + (a>>2));
  }
};
struct UnaryRegistry {
  std::unordered_map<OpKey, OutFn,      OpKeyHash> out_map;
  std::unordered_map<OpKey, InFn,       OpKeyHash> in_map;
  std::unordered_map<OpKey, OutParamFn, OpKeyHash> outp_map;
  std::unordered_map<OpKey, InParamFn,  OpKeyHash> inp_map;
};
static inline UnaryRegistry& registry() {
  static UnaryRegistry R;
  return R;
}
static inline void register_out (UnaryOp op, Dtype dt, OutFn f)      { registry().out_map [{op,dt}] = f; }
static inline void register_in  (UnaryOp op, Dtype dt, InFn f)       { registry().in_map  [{op,dt}] = f; }
static inline void register_outp(UnaryOp op, Dtype dt, OutParamFn f) { registry().outp_map[{op,dt}] = f; }
static inline void register_inp (UnaryOp op, Dtype dt, InParamFn f)  { registry().inp_map [{op,dt}] = f; }

// ===============================
// One-time registration
// ===============================
static inline void ensure_registered() {
  static std::once_flag once;
  std::call_once(once, []{
    // -------- Trig / Hyperbolic
    #define REG_TRIG(OP, KIND) \
      register_out(UnaryOp::OP, Dtype::Float32, +[](const void* a, void* b, size_t n){ \
        trig_detail::apply_unary_kernel<float,  trig_detail::UnaryKind::KIND>( \
          static_cast<const float*>(a), static_cast<float*>(b), n); }); \
      register_out(UnaryOp::OP, Dtype::Float64, +[](const void* a, void* b, size_t n){ \
        trig_detail::apply_unary_kernel<double, trig_detail::UnaryKind::KIND>( \
          static_cast<const double*>(a), static_cast<double*>(b), n); }); \
      register_in (UnaryOp::OP, Dtype::Float32, +[](void* p, size_t n){ \
        auto f = static_cast<float*>(p); \
        trig_detail::apply_unary_kernel<float,  trig_detail::UnaryKind::KIND>(f,f,n); }); \
      register_in (UnaryOp::OP, Dtype::Float64, +[](void* p, size_t n){ \
        auto d = static_cast<double*>(p); \
        trig_detail::apply_unary_kernel<double, trig_detail::UnaryKind::KIND>(d,d,n); });

    REG_TRIG(Sin,   Sin)   REG_TRIG(Cos,   Cos)   REG_TRIG(Tan,   Tan)
    REG_TRIG(Asin,  Asin)  REG_TRIG(Acos,  Acos)  REG_TRIG(Atan,  Atan)
    REG_TRIG(Sinh,  Sinh)  REG_TRIG(Cosh,  Cosh)  REG_TRIG(Tanh,  Tanh)
    REG_TRIG(Asinh, Asinh) REG_TRIG(Acosh, Acosh) REG_TRIG(Atanh, Atanh)
    #undef REG_TRIG

    // -------- Exp / Log (out-of-place; in-place = pass same ptr)
    register_out(UnaryOp::Exp,   Dtype::Float32, exp_log_detail::exp_float_kernel);
    register_out(UnaryOp::Exp,   Dtype::Float64, exp_log_detail::exp_double_kernel);
    register_in (UnaryOp::Exp,   Dtype::Float32, +[](void* p,size_t n){ exp_log_detail::exp_float_kernel  (p,p,n); });
    register_in (UnaryOp::Exp,   Dtype::Float64, +[](void* p,size_t n){ exp_log_detail::exp_double_kernel (p,p,n); });

    register_out(UnaryOp::Exp2,  Dtype::Float32, exp_log_detail::exp2_float_kernel);
    register_out(UnaryOp::Exp2,  Dtype::Float64, exp_log_detail::exp2_double_kernel);
    register_in (UnaryOp::Exp2,  Dtype::Float32, +[](void* p,size_t n){ exp_log_detail::exp2_float_kernel (p,p,n); });
    register_in (UnaryOp::Exp2,  Dtype::Float64, +[](void* p,size_t n){ exp_log_detail::exp2_double_kernel(p,p,n); });

    register_out(UnaryOp::Log,   Dtype::Float32, exp_log_detail::log_float_kernel);
    register_out(UnaryOp::Log,   Dtype::Float64, exp_log_detail::log_double_kernel);
    register_in (UnaryOp::Log,   Dtype::Float32, +[](void* p,size_t n){ exp_log_detail::log_float_kernel  (p,p,n); });
    register_in (UnaryOp::Log,   Dtype::Float64, +[](void* p,size_t n){ exp_log_detail::log_double_kernel (p,p,n); });

    register_out(UnaryOp::Log2,  Dtype::Float32, exp_log_detail::log2_float_kernel);
    register_out(UnaryOp::Log2,  Dtype::Float64, exp_log_detail::log2_double_kernel);
    register_in (UnaryOp::Log2,  Dtype::Float32, +[](void* p,size_t n){ exp_log_detail::log2_float_kernel (p,p,n); });
    register_in (UnaryOp::Log2,  Dtype::Float64, +[](void* p,size_t n){ exp_log_detail::log2_double_kernel(p,p,n); });

    register_out(UnaryOp::Log10, Dtype::Float32, exp_log_detail::log10_float_kernel);
    register_out(UnaryOp::Log10, Dtype::Float64, exp_log_detail::log10_double_kernel);
    register_in (UnaryOp::Log10, Dtype::Float32, +[](void* p,size_t n){ exp_log_detail::log10_float_kernel (p,p,n); });
    register_in (UnaryOp::Log10, Dtype::Float64, +[](void* p,size_t n){ exp_log_detail::log10_double_kernel(p,p,n); });

    // -------- Algebraic (all integer + float where defined)
    #define REG_ALG_IN(OP, FUNC) \
      register_in(UnaryOp::OP, Dtype::Float32, +[](void* p,size_t n){ alg_detail::FUNC<float >( (float*)p,n); }); \
      register_in(UnaryOp::OP, Dtype::Float64, +[](void* p,size_t n){ alg_detail::FUNC<double>( (double*)p,n); }); \
      register_in(UnaryOp::OP, Dtype::Int16 ,  +[](void* p,size_t n){ alg_detail::FUNC<int16_t>( (int16_t*)p,n); }); \
      register_in(UnaryOp::OP, Dtype::Int32 ,  +[](void* p,size_t n){ alg_detail::FUNC<int32_t>( (int32_t*)p,n); }); \
      register_in(UnaryOp::OP, Dtype::Int64 ,  +[](void* p,size_t n){ alg_detail::FUNC<int64_t>( (int64_t*)p,n); });

    #define REG_ALG_OUT(OP, FUNC2) \
      register_out(UnaryOp::OP, Dtype::Float32, +[](const void* a,void* b,size_t n){ alg_detail::FUNC2<float >( (const float*)a,(float*)b,n); }); \
      register_out(UnaryOp::OP, Dtype::Float64, +[](const void* a,void* b,size_t n){ alg_detail::FUNC2<double>( (const double*)a,(double*)b,n); }); \
      register_out(UnaryOp::OP, Dtype::Int16 ,  +[](const void* a,void* b,size_t n){ alg_detail::FUNC2<int16_t>( (const int16_t*)a,(int16_t*)b,n); }); \
      register_out(UnaryOp::OP, Dtype::Int32 ,  +[](const void* a,void* b,size_t n){ alg_detail::FUNC2<int32_t>( (const int32_t*)a,(int32_t*)b,n); }); \
      register_out(UnaryOp::OP, Dtype::Int64 ,  +[](const void* a,void* b,size_t n){ alg_detail::FUNC2<int64_t>( (const int64_t*)a,(int64_t*)b,n); });

    REG_ALG_IN (Square,     square_impl)      REG_ALG_OUT(Square,     square2_impl)
    REG_ALG_IN (Neg,        negate_impl)      REG_ALG_OUT(Neg,        negate2_impl)
    REG_ALG_IN (Abs,        abs_impl)         REG_ALG_OUT(Abs,        abs2_impl)
    REG_ALG_IN (Sign,       sign_impl)        REG_ALG_OUT(Sign,       sign2_impl)
    REG_ALG_IN (Reciprocal, reciprocal_impl)  REG_ALG_OUT(Reciprocal, reciprocal2_impl)

    // sqrt: float/double only
    register_in (UnaryOp::Sqrt, Dtype::Float32, +[](void* p,size_t n){ alg_detail::sqrt_impl<float >( (float*)p,n); });
    register_in (UnaryOp::Sqrt, Dtype::Float64, +[](void* p,size_t n){ alg_detail::sqrt_impl<double>( (double*)p,n); });
    register_out(UnaryOp::Sqrt, Dtype::Float32, +[](const void* a,void* b,size_t n){ alg_detail::sqrt2_impl<float >( (const float*)a,(float*)b,n); });
    register_out(UnaryOp::Sqrt, Dtype::Float64, +[](const void* a,void* b,size_t n){ alg_detail::sqrt2_impl<double>( (const double*)a,(double*)b,n); });

    // Power (parametric)
    register_inp (UnaryOp::Power, Dtype::Float32, +[](void* p,size_t n,double e){ alg_detail::power_impl<float >( (float*)p,n,e); });
    register_inp (UnaryOp::Power, Dtype::Float64, +[](void* p,size_t n,double e){ alg_detail::power_impl<double>( (double*)p,n,e); });
    register_outp(UnaryOp::Power, Dtype::Float32, +[](const void* a,void* b,size_t n,double e){ alg_detail::power2_impl<float >( (const float*)a,(float*)b,n,e); });
    register_outp(UnaryOp::Power, Dtype::Float64, +[](const void* a,void* b,size_t n,double e){ alg_detail::power2_impl<double>( (const double*)a,(double*)b,n,e); });

    // Optional integer power
    register_inp (UnaryOp::Power, Dtype::Int16, +[](void* p,size_t n,double e){ alg_detail::power_impl<int16_t>( (int16_t*)p,n,e); });
    register_inp (UnaryOp::Power, Dtype::Int32, +[](void* p,size_t n,double e){ alg_detail::power_impl<int32_t>( (int32_t*)p,n,e); });
    register_inp (UnaryOp::Power, Dtype::Int64, +[](void* p,size_t n,double e){ alg_detail::power_impl<int64_t>( (int64_t*)p,n,e); });
    register_outp(UnaryOp::Power, Dtype::Int16, +[](const void* a,void* b,size_t n,double e){ alg_detail::power2_impl<int16_t>( (const int16_t*)a,(int16_t*)b,n,e); });
    register_outp(UnaryOp::Power, Dtype::Int32, +[](const void* a,void* b,size_t n,double e){ alg_detail::power2_impl<int32_t>( (const int32_t*)a,(int32_t*)b,n,e); });
    register_outp(UnaryOp::Power, Dtype::Int64, +[](const void* a,void* b,size_t n,double e){ alg_detail::power2_impl<int64_t>( (const int64_t*)a,(int64_t*)b,n,e); });

    #undef REG_ALG_IN
    #undef REG_ALG_OUT
  });
}

// ===============================
// Core dispatch (out-of-place / in-place)
// ===============================
static inline Tensor unary_impl_out(const Tensor& x, UnaryOp op) {
  ensure_registered();
  const size_t n = x.numel();
  const Dtype  in_dt  = x.dtype();
  const Dtype  out_dt = promoted_dtype_for(x, op);

  // f16/bf16 bounce to f32
  if (is_f16(in_dt)) {
    const bool bf16 = (in_dt==Dtype::Bfloat16);
    std::vector<float> in_f(n), out_f(n);
    f16_to_f32(reinterpret_cast<const uint16_t*>(x.data()), in_f.data(), n, bf16);
    auto it = registry().out_map.find({op, Dtype::Float32});
    if (it==registry().out_map.end()) throw std::runtime_error("No Float32 kernel for op");
    it->second(in_f.data(), out_f.data(), n);
    Tensor y(Shape{x.shape()}, TensorOptions{in_dt, x.device(), x.requires_grad()});
    f32_to_f16(out_f.data(), reinterpret_cast<uint16_t*>(y.data()), n, bf16);
    return y;
  }

  // integer → float promotion for selected ops
  if (in_dt!=out_dt) {
    std::vector<float> in_f(n), out_f(n);
    if (in_dt==Dtype::Int16) {
      auto p=reinterpret_cast<const int16_t*>(x.data()); for(size_t i=0;i<n;++i) in_f[i]=static_cast<float>(p[i]);
    } else if (in_dt==Dtype::Int32) {
      auto p=reinterpret_cast<const int32_t*>(x.data()); for(size_t i=0;i<n;++i) in_f[i]=static_cast<float>(p[i]);
    } else {
      auto p=reinterpret_cast<const int64_t*>(x.data()); for(size_t i=0;i<n;++i) in_f[i]=static_cast<float>(p[i]);
    }
    auto it = registry().out_map.find({op, Dtype::Float32});
    if (it==registry().out_map.end()) throw std::runtime_error("No Float32 kernel for op");
    it->second(in_f.data(), out_f.data(), n);
    Tensor y(Shape{x.shape()}, TensorOptions{out_dt, x.device(), x.requires_grad()});
    std::memcpy(y.data(), out_f.data(), n*sizeof(float));
    return y;
  }

  // direct
  auto it = registry().out_map.find({op, in_dt});
  if (it==registry().out_map.end()) throw std::runtime_error("No kernel for op/dtype");
  Tensor y(Shape{x.shape()}, TensorOptions{out_dt, x.device(), x.requires_grad()});
  it->second(x.data(), y.data(), n);
  return y;
}

static inline void unary_impl_in(Tensor& x, UnaryOp op) {
  ensure_registered();
  const size_t n = x.numel();
  const Dtype  dt = x.dtype();

  if (is_f16(dt)) {
    const bool bf16 = (dt==Dtype::Bfloat16);
    std::vector<float> buf(n);
    f16_to_f32(reinterpret_cast<const uint16_t*>(x.data()), buf.data(), n, bf16);
    auto it = registry().in_map.find({op, Dtype::Float32});
    if (it==registry().in_map.end()) throw std::runtime_error("No Float32 inplace kernel for op");
    it->second(buf.data(), n);
    f32_to_f16(buf.data(), reinterpret_cast<uint16_t*>(x.data()), n, bf16);
    return;
  }

  if (op_promotes_ints(op) && is_int(dt)) {
    // in-place with promotion -> replace storage with promoted result
    x = unary_impl_out(x, op);
    return;
  }

  auto it = registry().in_map.find({op, dt});
  if (it==registry().in_map.end()) throw std::runtime_error("No inplace kernel for op/dtype");
  it->second(x.data(), n);
}

// Parametric
static inline Tensor unary_impl_out_param(const Tensor& x, UnaryOp op, double p) {
  ensure_registered();
  if (!op_needs_param(op)) return unary_impl_out(x, op);

  const size_t n = x.numel();
  const Dtype  dt = x.dtype();

  if (is_f16(dt)) {
    const bool bf16 = (dt==Dtype::Bfloat16);
    std::vector<float> in_f(n), out_f(n);
    f16_to_f32(reinterpret_cast<const uint16_t*>(x.data()), in_f.data(), n, bf16);
    auto it = registry().outp_map.find({op, Dtype::Float32});
    if (it==registry().outp_map.end()) throw std::runtime_error("No Float32 param kernel");
    it->second(in_f.data(), out_f.data(), n, p);
    Tensor y(Shape{x.shape()}, TensorOptions{dt, x.device(), x.requires_grad()});
    f32_to_f16(out_f.data(), reinterpret_cast<uint16_t*>(y.data()), n, bf16);
    return y;
  }

  auto it = registry().outp_map.find({op, dt});
  if (it==registry().outp_map.end()) throw std::runtime_error("No param kernel for op/dtype");
  Tensor y(Shape{x.shape()}, TensorOptions{dt, x.device(), x.requires_grad()});
  it->second(x.data(), y.data(), n, p);
  return y;
}

static inline void unary_impl_in_param(Tensor& x, UnaryOp op, double p) {
  ensure_registered();
  if (!op_needs_param(op)) { unary_impl_in(x, op); return; }

  const size_t n = x.numel();
  const Dtype  dt = x.dtype();

  if (is_f16(dt)) {
    const bool bf16 = (dt==Dtype::Bfloat16);
    std::vector<float> buf(n);
    f16_to_f32(reinterpret_cast<const uint16_t*>(x.data()), buf.data(), n, bf16);
    auto it = registry().inp_map.find({op, Dtype::Float32});
    if (it==registry().inp_map.end()) throw std::runtime_error("No Float32 param inplace kernel");
    it->second(buf.data(), n, p);
    f32_to_f16(buf.data(), reinterpret_cast<uint16_t*>(x.data()), n, bf16);
    return;
  }

  auto it = registry().inp_map.find({op, dt});
  if (it==registry().inp_map.end()) throw std::runtime_error("No param inplace kernel for op/dtype");
  it->second(x.data(), n, p);
}

// ===============================
// Public API
// ===============================
Tensor unary(const Tensor& x, UnaryOp op) { return unary_impl_out(x, op); }
void   unary_(Tensor& x, UnaryOp op)      { unary_impl_in(x, op); }
Tensor unary_param(const Tensor& x, UnaryOp op, double p) { return unary_impl_out_param(x, op, p); }
void   unary_param_(Tensor& x, UnaryOp op, double p)      { unary_impl_in_param(x, op, p); }

// Convenience wrappers
#define DEF_WRAP(name, OP) \
  Tensor name(const Tensor& x){ return unary(x, UnaryOp::OP); } \
  void   name##_(Tensor& x){ unary_(x, UnaryOp::OP); }

DEF_WRAP(sin,   Sin)   DEF_WRAP(cos,   Cos)   DEF_WRAP(tan,   Tan)
DEF_WRAP(asin,  Asin)  DEF_WRAP(acos,  Acos)  DEF_WRAP(atan,  Atan)
DEF_WRAP(sinh,  Sinh)  DEF_WRAP(cosh,  Cosh)  DEF_WRAP(tanh,  Tanh)
DEF_WRAP(asinh, Asinh) DEF_WRAP(acosh, Acosh) DEF_WRAP(atanh, Atanh)
DEF_WRAP(exp,   Exp)   DEF_WRAP(exp2,  Exp2)
DEF_WRAP(log,   Log)   DEF_WRAP(log2,  Log2)  DEF_WRAP(log10, Log10)
DEF_WRAP(square, Square)
DEF_WRAP(square_root, Sqrt)
DEF_WRAP(negator, Neg)
DEF_WRAP(absolute, Abs)
DEF_WRAP(sign, Sign)
DEF_WRAP(reciprocal, Reciprocal)
#undef DEF_WRAP

Tensor power(const Tensor& x, int e){ return unary_param(x, UnaryOp::Power, double(e)); }
void   power_(Tensor& x, int e){ unary_param_(x, UnaryOp::Power, double(e)); }

}