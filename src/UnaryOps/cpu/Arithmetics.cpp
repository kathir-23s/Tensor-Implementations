#include "ops/UnaryOps/Arithmetics.h"
#include "core/Tensor.h"
#include "device/Device.h"
#include "device/DeviceCore.h"
#include "ops/helpers/arith.hpp"
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// OUT-OF-PLACE OPERATIONS
// ============================================================================

Tensor square(const Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return square_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return square_out_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor sqrt(const Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return square_root_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return square_root_out_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor neg(const Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return negator_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return negator_out_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor abs(const Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return absolute_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return absolute_out_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor sign(const Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return sign_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return sign_out_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor reciprocal(const Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return reciprocal_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return reciprocal_out_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}


Tensor pow(const Tensor& input, int exponent, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return power_out_cpu_wrap(input, exponent);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return power_out_gpu_wrap(input, exponent, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor pow(const Tensor& input, float exponent, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return power_out_cpu_wrap(input, exponent);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return power_out_gpu_wrap(input, exponent, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor pow(const Tensor& input, double exponent, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        return power_out_cpu_wrap(input, exponent);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return power_out_gpu_wrap(input, exponent, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}
// ============================================================================
// IN-PLACE OPERATIONS
// ============================================================================

void square_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        square_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        square_in_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void sqrt_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        square_root_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        square_root_in_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void neg_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        negator_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        negator_in_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void abs_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        absolute_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        absolute_in_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void sign_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        sign_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        sign_in_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void reciprocal_(Tensor& input, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        reciprocal_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        reciprocal_in_gpu_wrap(input, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void pow_(Tensor& input, int exponent, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        power_in_cpu_wrap(input, exponent);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        power_in_gpu_wrap(input, exponent, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void pow_(Tensor& input, float exponent, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        power_in_cpu_wrap(input, exponent);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        power_in_gpu_wrap(input, exponent, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void pow_(Tensor& input, double exponent, cudaStream_t stream) {//✨✨✨
    if (input.is_cpu()) {
        power_in_cpu_wrap(input, exponent);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        power_in_gpu_wrap(input, exponent, stream);//✨✨✨
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}
} // namespace OwnTensor