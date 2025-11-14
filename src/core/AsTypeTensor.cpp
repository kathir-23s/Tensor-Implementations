#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/DeviceTransfer.h"

namespace OwnTensor
{
    Tensor Tensor::as_type(Dtype new_dtype) const {
        // Edge Case: If types are the same, just return a clone
        if (new_dtype == this->dtype_) {
            return this->clone();
        }

        // 1. Create the destination tensor on the SAME device
        Tensor new_tensor(this->shape_, TensorOptions{new_dtype, this->device_});

        // 2. Get element count
        const size_t n = this->numel();

        // ✅ 3. Handle CUDA tensors differently
        if (this->is_cuda()) {
            #ifdef WITH_CUDA
                // Strategy: Copy to CPU, convert, copy back to GPU
                
                // Step 3a: Copy source data from GPU → CPU
                Tensor cpu_source = this->to_cpu();
                
                // Step 3b: Convert on CPU
                const auto* src_untyped_ptr = cpu_source.data_ptr_.get();
                std::vector<uint8_t> cpu_converted(n * Tensor::dtype_size(new_dtype));
                auto* dst_untyped_ptr = cpu_converted.data();
                
                // Nested dispatch for type conversion (CPU)
                dispatch_by_dtype(this->dtype_, [&](auto src_type_placeholder) {
                    using SrcType = decltype(src_type_placeholder);
                    const auto* src_data = reinterpret_cast<const SrcType*>(src_untyped_ptr);

                    dispatch_by_dtype(new_dtype, [&](auto dst_type_placeholder) {
                        using DstType = decltype(dst_type_placeholder);
                        auto* dst_data = reinterpret_cast<DstType*>(dst_untyped_ptr);

                        // The core conversion loop
                        for (size_t i = 0; i < n; ++i) {
                            dst_data[i] = static_cast<DstType>(src_data[i]);
                        }
                    });
                });
                
                // Step 3c: Copy converted data from CPU → GPU
                device::copy_memory(
                    new_tensor.data_ptr_.get(), this->device_.device,  // GPU destination
                    cpu_converted.data(), Device::CPU,                  // CPU source
                    cpu_converted.size()
                );
            #else
                throw std::runtime_error("CUDA support not compiled");
            #endif
        } else {
            // ✅ 4. Handle CPU tensors (original logic)
            const auto* src_untyped_ptr = this->data_ptr_.get();
            auto* dst_untyped_ptr = new_tensor.data_ptr_.get();

            // Nested dispatch for type conversion
            dispatch_by_dtype(this->dtype_, [&](auto src_type_placeholder) {
                using SrcType = decltype(src_type_placeholder);
                const auto* src_data = reinterpret_cast<const SrcType*>(src_untyped_ptr);

                dispatch_by_dtype(new_dtype, [&](auto dst_type_placeholder) {
                    using DstType = decltype(dst_type_placeholder);
                    auto* dst_data = reinterpret_cast<DstType*>(dst_untyped_ptr);

                    // The core conversion loop
                    for (size_t i = 0; i < n; ++i) {
                        dst_data[i] = static_cast<DstType>(src_data[i]);
                    }
                });
            });
        }

        // 5. Return the newly created tensor
        return new_tensor;
    }
}