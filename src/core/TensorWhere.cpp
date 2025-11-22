#include "core/Tensor.h"
#include "ops/helpers/ConditionalOps.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include "ops/helpers/ConversionKernels.cuh"
#endif

namespace OwnTensor
{

            // Simple type promotion for where operation
        static Dtype promote_dtypes_internal(Dtype a, Dtype b) {
            if (a == b) return a;
            
            // Promotion hierarchy: Float64 > Float32 > Int64 > Int32 > Int16
            auto rank = [](Dtype d) -> int {
                switch(d) {
                    case Dtype::Float64: return 5;
                    case Dtype::Float32: return 4;
                    case Dtype::Int64: return 3;
                    case Dtype::Int32: return 2;
                    case Dtype::Int16: return 1;
                    case Dtype::Float16: return 4;
                    case Dtype::Bfloat16: return 4;
                    default: return 0;
                }
            };
            
            return (rank(a) > rank(b)) ? a : b;
        }
    Tensor Tensor::to_bool() const {
        Tensor result({this->shape()}, TensorOptions()
            .with_dtype(Dtype::Bool)
            .with_device(this->device()));  // Preserve device
        
        if (this->is_cpu()) {
            // CPU path - use existing OpenMP code
            dispatch_by_dtype(this->dtype(), [&](auto T_val) {
                using T = decltype(T_val);
                const T* src = this->data<T>();
                bool* dst = result.data<bool>();
                
                #pragma omp parallel for
                for (size_t i = 0; i < this->numel(); ++i) {
                    dst[i] = (src[i] != T(0));
                }
            });
        }
    #ifdef WITH_CUDA
        else if (this->is_cuda()) {
            dispatch_by_dtype(this->dtype(), [&](auto T_val) {
                using T = decltype(T_val);
                const T* src = this->data<T>();
                bool* dst = result.data<bool>();
                
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                
                // Launch conversion kernel
                convert_to_bool_cuda<T>(src, dst, this->numel(), stream);
                
                // ✅ Synchronization is ALREADY in convert_to_bool_cuda
                // No need to sync again here (but it doesn't hurt)
            });
        }  // ✅ ADD THIS CLOSING BRACE
    #endif
        else {
            throw std::runtime_error("to_bool: Unknown device type");
        }
        
        return result;
    }
        // Simple shape broadcasting check (for now, require exact match)
        static bool shapes_match(const Shape& a, const Shape& b) {
            if (a.dims.size() != b.dims.size()) return false;
            for (size_t i = 0; i < a.dims.size(); ++i) {
                if (a.dims[i] != b.dims[i]) return false;
            }
            return true;
        }
    // ============================================================================
    // WHERE Implementation
    // ============================================================================

    // Main where implementation - simplified version without broadcasting
    Tensor Tensor::where(const Tensor& condition, const Tensor& input, const Tensor& other) {
    // Validate condition dtype
        if (condition.dtype_ != Dtype::Int32 && condition.dtype_ != Dtype::Int64 ) {
            throw std::invalid_argument("Condition must be Int32 or Int64 dtype");
        }
        
        // Validate same device
        if (condition.device_.device != input.device_.device || 
            input.device_.device != other.device_.device) {
            throw std::invalid_argument("All tensors must be on same device");
        }
        
        // Validate same shape
        if (!shapes_match(condition.shape_, input.shape_) || 
            !shapes_match(input.shape_, other.shape_)) {
            throw std::invalid_argument("All tensors must have same shape");
        }
        
        // Determine output dtype
        Dtype output_dtype = promote_dtypes_internal(input.dtype_, other.dtype_);
        
        // Create output tensor
        Tensor result(input.shape_, output_dtype, input.device_);
        
        // Dispatch to CPU or CUDA backend
        if (condition.is_cpu()) {
            cpu_where(condition, input, other, result);
        } else {
    #ifdef WITH_CUDA
            cuda_where(condition, input, other, result);
    #else
            throw std::runtime_error("CUDA support not compiled");
    #endif
        }
        
        return result;
    }

    // Scalar overload - requires Tensor::full() implementation
    Tensor Tensor::where(const Tensor& condition, float input_scalar, const Tensor& other) {
        Tensor input_tensor = Tensor::full(condition.shape(), 
                                        TensorOptions().with_dtype(other.dtype()).with_device(condition.device()), 
                                        input_scalar);
        return where(condition, input_tensor, other);
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& input, float other_scalar) {
        Tensor other_tensor = Tensor::full(condition.shape(), 
                                        TensorOptions().with_dtype(input.dtype()).with_device(condition.device()), 
                                        other_scalar);
        return where(condition, input, other_tensor);
    }

    Tensor Tensor::where(const Tensor& condition, float input_scalar, float other_scalar) {
        Tensor input_tensor = Tensor::full(condition.shape(), 
                                        TensorOptions().with_dtype(Dtype::Float32).with_device(condition.device()), 
                                        input_scalar);
        Tensor other_tensor = Tensor::full(condition.shape(), 
                                        TensorOptions().with_dtype(Dtype::Float32).with_device(condition.device()), 
                                        other_scalar);
        
        return where(condition, input_tensor, other_tensor);
    }
}