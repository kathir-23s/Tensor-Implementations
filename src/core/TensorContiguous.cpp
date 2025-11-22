#include "core/Tensor.h"
#include "device/Allocator.h"
#include "device/AllocatorRegistry.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include "core/Views/contiguous_kernel.h"
#include "ops/helpers/ConversionKernels.cuh"
#endif


namespace OwnTensor
{
bool Tensor::is_contiguous() const
    {
        // Need to look into it
        // What it is and what's it for
        int64_t expected_stride = 1;
        const auto& dims = shape_.dims;
        const auto& strides = stride_.strides;
       
        for (int i = dims.size() - 1; i >= 0; --i)
        {
            if (strides[i] != expected_stride)
            {
                return false;
            }
            expected_stride *= dims[i];
        }
        return true;
    }

    Tensor Tensor::contiguous() const {
        // If already contiguous with zero offset, return a bytewise copy that owns data.
        // Returning a copy (not aliasing) keeps semantics clear and avoids alias bugs.
        if (is_contiguous() && storage_offset_ == 0) {
            Tensor out(shape_, dtype_, device_);
            Allocator* alloc = AllocatorRegistry::get_allocator(device_.device);
            // alloc->memcpy(out.data(), data(), nbytes());
            alloc->memcpy(out.data(), data(), nbytes(), is_cpu() ? cudaMemcpyHostToHost : cudaMemcpyDeviceToDevice);//✨✨✨
            return out;
        }

        // Allocate destination with row‑major layout on the same device
        Tensor out(shape_, dtype_, device_);
        Allocator* alloc = AllocatorRegistry::get_allocator(device_.device);

        const size_t bytes_per_elem = dtype_size(dtype_);
        const int64_t total_elems = static_cast<int64_t>(numel());
        const size_t D = shape_.dims.size();

        if (is_cpu()) {
            std::vector<int64_t> idx(D, 0);

            auto bump = [&](std::vector<int64_t>& v)->bool {
                for (int d = int(D) - 1; d >= 0; --d) {
                    if (++v[d] < shape_.dims[d]) return true;
                    v[d] = 0;
                }
                return false;
            };

            uint8_t* dst = reinterpret_cast<uint8_t*>(out.data());
            size_t write_pos = 0;

            do {
                // Compute element offset in elements: sum(idx[d] * stride[d])
                // DON'T add storage_offset here!
                int64_t elem_off = 0;
                for (size_t d = 0; d < D; ++d) {
                    elem_off += idx[d] * stride_.strides[d];
                }

                // data() already accounts for storage_offset, so just add elem_off
                const uint8_t* src_elem_ptr =
                    reinterpret_cast<const uint8_t*>(data())
                    + elem_off * bytes_per_elem;

                // alloc->memcpy(dst + write_pos, src_elem_ptr, bytes_per_elem);
                alloc->memcpy(dst + write_pos, src_elem_ptr, bytes_per_elem, cudaMemcpyHostToHost);//✨✨✨
                write_pos += bytes_per_elem;

            } while (bump(idx));

            return out;
        }
        #ifdef WITH_CUDA
            else if (is_cuda()) {
                cudaStream_t stream = 0;
                
                // *** CRITICAL FIX: Copy dims and strides to GPU memory first! ***
                int64_t* d_dims = nullptr;
                int64_t* d_strides = nullptr;
                
                cudaMalloc(&d_dims, D * sizeof(int64_t));
                cudaMalloc(&d_strides, D * sizeof(int64_t));
                
                cudaMemcpy(d_dims, shape_.dims.data(), D * sizeof(int64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_strides, stride_.strides.data(), D * sizeof(int64_t), cudaMemcpyHostToDevice);
                
                contiguous_strided_copy_cuda(
                    data(), out.data(), total_elems,
                    d_dims,      // ← GPU pointer
                    d_strides,   // ← GPU pointer  
                    static_cast<int32_t>(D),
                    0,
                    static_cast<int32_t>(bytes_per_elem),
                    stream
                );

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    cudaFree(d_dims);
                    cudaFree(d_strides);
                    throw std::runtime_error(std::string("contiguous kernel launch failed: ")
                                            + cudaGetErrorString(err));
                }
                
                // Synchronize and clean up
                // cudaDeviceSynchronize();//✨✨✨
                cudaFree(d_dims);
                cudaFree(d_strides);
                
                return out;
            }
            #endif
            else {
                throw std::runtime_error("Unknown device in Tensor::contiguous()");
            }
        }

    Tensor Tensor::clone() const
    {
        // Edge case: Empty tensor
        if (numel() == 0) {
            return Tensor(shape_, dtype_, device_);
        }
        
        // Edge case: Non-contiguous or has storage_offset - materialize first
        if (!is_contiguous() || storage_offset_ != 0) {
            try {
                Tensor src_contig = contiguous();  // Uses your contiguous_kernel.cu for GPU
                Tensor result(src_contig.shape_, dtype_, device_);
                
                Allocator* alloc = AllocatorRegistry::get_allocator(device_.device);
                // alloc->memcpy(result.data(), src_contig.data(), src_contig.nbytes());
                alloc->memcpy(result.data(), src_contig.data(), src_contig.nbytes(), is_cpu() ? cudaMemcpyHostToHost : cudaMemcpyDeviceToDevice);//✨✨✨

                return result;
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("clone failed (contiguous): ") + e.what());
            }
        }
        
        // Contiguous path: direct clone
        try {
            Tensor result(shape_, dtype_, device_);
            
            Allocator* alloc = AllocatorRegistry::get_allocator(device_.device);
            // alloc->memcpy(result.data(), data(), nbytes());
            alloc->memcpy(result.data(), data(), nbytes(), is_cpu() ? cudaMemcpyHostToHost : cudaMemcpyDeviceToDevice);//✨✨✨
            
            return result;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("clone failed: ") + e.what());
        }
    }

    Tensor& Tensor::copy_(const Tensor& src)
    {
        // Edge case: Self-copy is no-op
        if (this == &src || data() == src.data()) return *this;
        // Edge case: Empty tensor
        if (numel() == 0 && src.numel() == 0) {
            return *this;
        }
        // Edge case: Size validation
        if (numel() != src.numel()) {
            throw std::runtime_error(
                "copy_: size mismatch. Destination has " + 
                std::to_string(numel()) + " elements but source has " + 
                std::to_string(src.numel())
            );
        }
        if (dtype_ != src.dtype_) {
            throw std::runtime_error("copy_: dtype mismatch");
        }
        if (numel() == 0) return *this;
        if (!is_contiguous() || storage_offset_ != 0) {
            throw std::runtime_error("copy_: destination must be contiguous");
        }
        
        // Materialize non-contiguous source
        const Tensor* src_ptr = &src;
        if (!src.is_contiguous() || src.storage_offset_ != 0) {
            Tensor src_contig = src.contiguous();
            src_ptr = &src_contig;
        }
        try {
            device::copy_memory(
                data(), device_.device,           // destination ptr and device
                src_ptr->data(), src_ptr->device_.device,  // source ptr and device
                nbytes()
            );
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("copy_ failed: ") + e.what());
        }
        
        return *this;
    }
}