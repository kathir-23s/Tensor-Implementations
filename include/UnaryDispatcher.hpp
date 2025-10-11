#pragma once
#include <functional>
#include <unordered_map>

// custom headers
#include "Tensor.h"
#include "fp16_bf16_convert.h"
#include "DtypeTraits.h"

enum class UnaryOp:uint8_t{
    // trigonometric + hyperbolic
    Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
    // exponentiation + logarithmic
    Exp, Exp2, Log, Log2, Log10,
    // Algebraic
    Square, Sqrt, Neg, Abs, Sign, Reciprocal,
    // parametric
    Power
};

enum class ExecutionMode:uint8_t {
    Out_of_Place,
    In_Place
};

using UnaryKernelFn = std::function<void(const void* in, void* out, size_t size, double p)>;

struct KernelKey {
    UnaryOp op;
    Dtype dtype;
    Device device;
    ExecutionMode mode;
    bool operator==(const KernelKey& other) const;
};

struct KernelKeyHash { size_t operator()(const KernelKey& key) const noexcept; };

class KernelRegistry {
public:
    static KernelRegistry& instance();
    void register_kernel(const KernelKey& key, UnaryKernelFn fn);
    UnaryKernelFn get_kernel(const KernelKey& key) const;
private:
    KernelRegistry() = default;
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;
    std::unordered_map<KernelKey, UnaryKernelFn, KernelKeyHash> registry_;
};

inline bool is_fp16(Dtype d){ return d == Dtype::Bfloat16 || d == Dtype::Float16; }

namespace trig{
    enum class Unary{
        Sin, Cos, Tan, Asin, Acos, Atan,
        Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
    };
    template <typename T, Unary U>
    void apply_unary_kernel(const T* in, T* out, size_t size);
}

namespace exp_log{
    enum class Unary{
        Exp, Exp2, Log, Log2, Log10
    };
    template <typename T, Unary U>
    void apply_unary_kernel(const T* in, T* out, size_t size);
}

namespace alg{
    enum class Unary{
        Square, Sqrt, Neg, Abs, Sign, Reciprocal, Power
    };
    template <typename T, Unary U>
    void apply_unary_kernel(const T* in, T* out, size_t size);
    
    template <typename T, Unary U>
    void apply_unary_kernel(const T* in, T* out, size_t size, double para);
}

namespace trig { template <typename T, trig::Unary U> void apply_unary_kernel(const T* in, T* out, size_t size); }
namespace exp_log { template <typename T, exp_log::Unary U> void apply_unary_kernel(const T* in, T* out, size_t size); }
namespace alg { 
    template <typename T, alg::Unary U> void apply_unary_kernel(const T* in, T* out, size_t size); 
    template <typename T, alg::Unary U> void apply_unary_kernel(const T* in, T* out, size_t size, double p); 
}

inline bool op_needs_param(UnaryOp op) { return op == UnaryOp::Power; }

// for debugging and logging purposes
inline std::string device_to_string(Device device){ return device == Device::CPU ? "CPU" : "GPU"; };

inline std::string dtype_to_string(Dtype dtype){
    switch(dtype){
        case Dtype::Int16: return "Int16"; break;
        case Dtype::Int32: return "int32"; break;
        case Dtype::Int64: return "Int64"; break;
        case Dtype::Bfloat16: return "Bfloat16"; break;
        case Dtype::Float16: return "Float16"; break;
        case Dtype::Float32: return "Float32"; break;
        case Dtype::Float64: return "Float64"; break;
        default: return "unknown dtype";
    }
}

inline std::string unaryop_to_string(UnaryOp op){
    switch(op){
        // trig functions
        case UnaryOp::Sin: return "Sin";
        case UnaryOp::Cos: return "Cos";
        case UnaryOp::Tan: return "Tan";
        case UnaryOp::Asin: return "Asin";
        case UnaryOp::Acos: return "Acos";
        case UnaryOp::Atan: return "Atan";
        case UnaryOp::Sinh: return "Sinh";
        case UnaryOp::Cosh: return "Cosh";
        case UnaryOp::Tanh: return "Tanh";
        case UnaryOp::Asinh: return "Asinh";
        case UnaryOp::Acosh: return "Acosh";
        case UnaryOp::Atanh: return "Atanh";

        // exp & log functions
        case UnaryOp::Exp: return "Exp";
        case UnaryOp::Exp2: return "Exp2";
        case UnaryOp::Log: return "Log";
        case UnaryOp::Log2: return "Log2";
        case UnaryOp::Log10: return "Log10";

        // alg functions
        case UnaryOp::Square: return "Square";
        case UnaryOp::Sqrt: return "Sqrt";
        case UnaryOp::Neg: return "Neg";
        case UnaryOp::Abs: return "Abs";
        case UnaryOp::Sign: return "Reciprocal";
        case UnaryOp::Power: return "Power";

        // default value
        default: return "unknown operation";
    }
};

inline std::string mode_to_string(ExecutionMode mode) {
    switch(mode) {
        case ExecutionMode::Out_of_Place: return "Out";
        case ExecutionMode::In_Place: return "In";
        default: return "unknown mode";
    }
}


// to check if the input needs to be promoted
inline bool op_promotes_int(UnaryOp op) {
    switch (op)
    {
    case UnaryOp::Sin:
    case UnaryOp::Cos:
    case UnaryOp::Tan:
    case UnaryOp::Asin:
    case UnaryOp::Acos:
    case UnaryOp::Atan:
    case UnaryOp::Sinh:
    case UnaryOp::Cosh:
    case UnaryOp::Tanh:
    case UnaryOp::Asinh:
    case UnaryOp::Acosh:
    case UnaryOp::Atanh:
    case UnaryOp::Exp:
    case UnaryOp::Exp2:
    case UnaryOp::Log:
    case UnaryOp::Log2:
    case UnaryOp::Log10:
        return true;
    
    default:
        return false;
    }
}

template <
    typename T,
    typename EnumType,
    EnumType OpEnum,
    void (*Func)(const T* in, T* out, size_t size)
>
struct AutoRegisterUnary {
    AutoRegisterUnary(UnaryOp op, Device device, ExecutionMode mode) {
        KernelKey key{op, DtypeFor<T>, device, mode};
        KernelRegistry::instance().register_kernel(
            key,
            +[](const void* in, void* out, size_t size, double /*p*/) {
                Func(static_cast<const T*>(in),
                     static_cast<T*>(out),
                     size);
            }
        );
    }
};


template <
    typename T,
    typename EnumType,
    EnumType OpEnum,
    void (*Func)(const T* in, T* out, size_t size, double para)
>
struct AutoRegisterUnaryParam {
    AutoRegisterUnaryParam(UnaryOp op, Device device, ExecutionMode mode) {
        KernelKey key{op, DtypeFor<T>, device, mode};
        KernelRegistry::instance().register_kernel(
            key,
            +[](const void* in, void* out, size_t size, double para) {
                Func(static_cast<const T*>(in),
                     static_cast<T*>(out),
                     size,
                     para);
            }
        );
    }
};

// --- New AutoRegister template for Dtype Promotion kernels ---

template <
    typename InputT,
    typename OutputT, // NEW: The type of the output tensor data
    typename EnumType,
    EnumType OpEnum,
    // The kernel function signature must match: InputT -> OutputT
    void (*Func)(const InputT* in, OutputT* out, size_t size) 
>
struct AutoRegisterUnaryPromote {
    AutoRegisterUnaryPromote(UnaryOp op, Device device) {
        // CRITICAL: The KernelKey uses the INPUT Dtype
        KernelKey key{op, DtypeFor<InputT>, device, ExecutionMode::Out_of_Place}; 
        
        // This is the function pointer that gets registered
        KernelRegistry::instance().register_kernel(
            key,
            +[](const void* in, void* out, size_t size, double /*p*/) {
                // The lambda wrapper handles the Dtype casting for both input and output
                Func(static_cast<const InputT*>(in),
                     static_cast<OutputT*>(out),
                     size);
            }
        );
    }
};
