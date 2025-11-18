#include "core/Tensor.h"
#include "device/Device.h"
#include "device/DeviceCore.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/helpers/exp_log.hpp"
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// High-Level API - Out-of-Place Operations
// ============================================================================
Tensor exp(const Tensor& input, cudaStream_t stream) {//✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return exp_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return exp_out_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for exp");
    }
}

Tensor exp2(const Tensor& input, cudaStream_t stream) {//✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return exp2_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return exp2_out_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for exp2");
    }
}

Tensor log(const Tensor& input, cudaStream_t stream) {//✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return log_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return log_out_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for log");
    }
}

Tensor log2(const Tensor& input, cudaStream_t stream) {//✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return log2_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return log2_out_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for log2");
    }
}

Tensor log10(const Tensor& input, cudaStream_t stream) {//✨✨✨
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        return log10_out_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        return log10_out_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for log10");
    }
}

// ============================================================================
// High-Level API - In-Place Operations
// ============================================================================
void exp_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"exp inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        exp_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        exp_in_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for exp_");
    }
}

void exp2_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"exp2 inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        exp2_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        exp2_in_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for exp2_");
    }
}

void log_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"log inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        log_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        log_in_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for log_");
    }
}

void log2_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"log2 inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        log2_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        log2_in_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for log2_");
    }
}

void log10_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.dtype() == Dtype::Bool) {
        throw std::runtime_error(
            "NotImplementedError: \"log10 inplace\" not implemented for 'Bool'"
        );
    }
    const auto& dev = input.device();
    if (dev.is_cpu()) {
        log10_in_cpu_wrap(input);
    } else if (dev.is_cuda()) {
        log10_in_gpu_wrap(input, stream);//✨✨✨
    } else {
        throw std::runtime_error("Unsupported device for log10_");
    }
}

} // namespace OwnTensor