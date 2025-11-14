#include "core/Tensor.h"
#include "device/Device.h"
#include "device/DeviceCore.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/helpers/Trigonometry.hpp"
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// High-Level API - Out-of-Place Trigonometric Operations
// ============================================================================

// Basic trigonometric functions
Tensor sin(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return sin_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return sin_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for sin");
    }
}

Tensor cos(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return cos_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return cos_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for cos");
    }
}

Tensor tan(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return tan_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return tan_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for tan");
    }
}

// Inverse trigonometric functions
Tensor asin(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return asin_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return asin_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for asin");
    }
}

Tensor acos(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return acos_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return acos_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for acos");
    }
}

Tensor atan(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return atan_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return atan_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for atan");
    }
}

// Hyperbolic functions
Tensor sinh(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return sinh_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return sinh_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for sinh");
    }
}

Tensor cosh(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return cosh_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return cosh_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for cosh");
    }
}

Tensor tanh(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return tanh_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return tanh_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for tanh");
    }
}

// Inverse hyperbolic functions
Tensor asinh(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return asinh_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return asinh_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for asinh");
    }
}

Tensor acosh(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return acosh_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return acosh_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for acosh");
    }
}

Tensor atanh(const Tensor& input, cudaStream_t stream) { //✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return atanh_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return atanh_out_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for atanh");
    }
}

// ============================================================================
// High-Level API - In-Place Trigonometric Operations
// ============================================================================

// Basic trigonometric functions
void sin_(Tensor& input, cudaStream_t stream) { //✨✨✨
  if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"Sin inplace\" not implemented for 'Bool'"
        );
    } 
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        sin_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        sin_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for sin_");
    }
}

void cos_(Tensor& input, cudaStream_t stream) { //✨✨✨
   if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"cos inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        cos_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        cos_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for cos_");
    }
}

void tan_(Tensor& input, cudaStream_t stream) { //✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"tan inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        tan_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        tan_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for tan_");
    }
}

// Inverse trigonometric functions
void asin_(Tensor& input, cudaStream_t stream) { //✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"asin inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        asin_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        asin_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for asin_");
    }
}

void acos_(Tensor& input, cudaStream_t stream) { //✨✨✨
   if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"acos inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        acos_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        acos_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for acos_");
    }
}

void atan_(Tensor& input, cudaStream_t stream) { //✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"atan inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        atan_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        atan_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for atan_");
    }
}

// Hyperbolic functions
void sinh_(Tensor& input, cudaStream_t stream) { //✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"sinh inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        sinh_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        sinh_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for sinh_");
    }
}

void cosh_(Tensor& input, cudaStream_t stream) { //✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"cosh inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        cosh_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        cosh_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for cosh_");
    }
}

void tanh_(Tensor& input, cudaStream_t stream) { //✨✨✨
   if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"tanh inplace\" not implemented for 'Bool'"
        );
    } 
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        tanh_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        tanh_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for tanh_");
    }
}

// Inverse hyperbolic functions
void asinh_(Tensor& input, cudaStream_t stream) { //✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"asinh inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        asinh_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        asinh_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for asinh_");
    }
}

void acosh_(Tensor& input, cudaStream_t stream) { //✨✨✨
   if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"acosh inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        acosh_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        acosh_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for acosh_");
    }
}

void atanh_(Tensor& input, cudaStream_t stream) { //✨✨✨
   if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"atanh inplace\" not implemented for 'Bool'"
        );
    } 
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        atanh_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        atanh_in_gpu_wrap(input, stream); //✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for atanh_");
    }
}

} // namespace OwnTensor