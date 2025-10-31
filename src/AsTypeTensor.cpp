#include "core/Tensor.h"
#include "core/TensorDispatch.h"

namespace OwnTensor
{
    Tensor Tensor::as_type(Dtype new_dtype) const {
    // Edge Case: If the types are the same, just return a deep copy.
    if (new_dtype == this->dtype_) {
        // Assuming you have a clone() method. If not, this is a simple implementation.
        Tensor new_tensor(this->shape_, TensorOptions{this->dtype_, this->device_});
        std::memcpy(new_tensor.data_ptr_.get(), this->data_ptr_.get(), this->allocated_bytes());
        return new_tensor;
    }

    // 1. Create the destination tensor.
    Tensor new_tensor(this->shape_, TensorOptions{new_dtype, this->device_});

    // 2. Get pointers and element count.
    const size_t n = this->numel();
    const auto* src_untyped_ptr = this->data_ptr_.get();
    auto* dst_untyped_ptr = new_tensor.data_ptr_.get();

    // 3. Use nested dispatch.
    // The outer dispatch resolves the source type.
    dispatch_by_dtype(this->dtype_, [&](auto src_type_placeholder) { // CHANGED: Removed the '*' from the placeholder
        
        // CHANGED: Use decltype directly on the value placeholder
        using SrcType = decltype(src_type_placeholder);
        
        const auto* src_data = reinterpret_cast<const SrcType*>(src_untyped_ptr);

        // The inner dispatch resolves the destination type.
        dispatch_by_dtype(new_dtype, [&](auto dst_type_placeholder) { // CHANGED: Removed the '*' from the placeholder
            
            // CHANGED: Use decltype directly on the value placeholder
            using DstType = decltype(dst_type_placeholder);

            auto* dst_data = reinterpret_cast<DstType*>(dst_untyped_ptr);

            // 4. The core conversion loop (this part is unchanged).
            for (size_t i = 0; i < n; ++i) {
                dst_data[i] = static_cast<DstType>(src_data[i]);
            }
        });
    });

    // 5. Return the newly created tensor.
    return new_tensor;
}
}