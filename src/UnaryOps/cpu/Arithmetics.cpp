#include "ops/UnaryOps/Arithmetics.h"
#include "Tensor.h"
#include "device/Device.h"
#include "ops/helpers/arith.hpp"
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// OUT-OF-PLACE OPERATIONS
// ============================================================================

Tensor square(const Tensor& input) {
    if (input.is_cpu()) {
        return square_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return square_out_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor square_root(const Tensor& input) {
    if (input.is_cpu()) {
        return square_root_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return square_root_out_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor negator(const Tensor& input) {
    if (input.is_cpu()) {
        return negator_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return negator_out_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor absolute(const Tensor& input) {
    if (input.is_cpu()) {
        return absolute_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return absolute_out_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor sign(const Tensor& input) {
    if (input.is_cpu()) {
        return sign_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return sign_out_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

Tensor reciprocal(const Tensor& input) {
    if (input.is_cpu()) {
        return reciprocal_out_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        return reciprocal_out_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

// ============================================================================
// IN-PLACE OPERATIONS
// ============================================================================

void square_(Tensor& input) {
    if (input.is_cpu()) {
        square_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        square_in_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void square_root_(Tensor& input) {
    if (input.is_cpu()) {
        square_root_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        square_root_in_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void negator_(Tensor& input) {
    if (input.is_cpu()) {
        negator_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        negator_in_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void absolute_(Tensor& input) {
    if (input.is_cpu()) {
        absolute_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        absolute_in_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void sign_(Tensor& input) {
    if (input.is_cpu()) {
        sign_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        sign_in_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

void reciprocal_(Tensor& input) {
    if (input.is_cpu()) {
        reciprocal_in_cpu_wrap(input);
    }
#ifdef WITH_CUDA
    else if (input.is_cuda()) {
        reciprocal_in_gpu_wrap(input);
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type");
    }
}

} // namespace OwnTensor