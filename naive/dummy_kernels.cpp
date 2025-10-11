// In dummy_kernels.cpp or similar file, or before the registration macro in unary_registry.cpp
#include <iostream>
#include <omp.h>
#include <cmath>
#include "../include/UnaryDispatcher.hpp"

// 1. Provide the necessary namespace and template definition
namespace trig {
    template <typename T, Unary U>
    void apply_unary_kernel(const T* in, T* out, size_t size) {
        // This is a dummy body. It doesn't actually compute Sin, 
        // but it satisfies the linker and lets the registration succeed.
        // We can add a print statement here to confirm the kernel *could* be called later.
        std::cout << "--- DUMMY KERNEL CALLED: " << size << " elements ---\n";
        // Optionally, copy input to output for a "no-op" result
        if (in != out) {
            // For a float kernel, this would be std::memcpy(out, in, size * sizeof(T));
        }
    }
    
    // Explicitly instantiate the template for the type used in the macro
    // This may be needed if the definition is in a separate file.
    template void apply_unary_kernel<float, Unary::Sin>(const float* in, float* out, size_t size);
}

namespace exp_log {
    template <typename T, Unary U>
    void apply_unary_kernel(const T* in, T* out, size_t size) {
        #pragma omp parallel for  // 2. Parallelize this loop with OpenMP
        for (size_t i = 0; i < size; ++i) {
            out[i] = std::exp(in[i]);
        }
        if (in != out) {
            // For a float kernel, this would be std::memcpy(out, in, size * sizeof(T));
        }
    }
    
    // Explicitly instantiate the template for the type used in the macro
    // This may be needed if the definition is in a separate file.
    template void apply_unary_kernel<float, Unary::Exp>(const float* in, float* out, size_t size);
}