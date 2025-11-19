#pragma once

#include <vector>
#include <memory>
#include "device/Device.h"
#include "dtype/Dtype.h"
#include "dtype/Types.h"

namespace OwnTensor
{
    // ########################################################################
    // Custom Type Definitions
    // ########################################################################

    // Shape and stride
    struct Shape
    {
        std::vector<int64_t> dims;
         // Define the equality operator for Shape objects
        bool operator==(const Shape& other) const {
            return dims == other.dims; // This uses the std::vector::operator==
        }

        // Optionally, define the inequality operator explicitly (less common, usually implicit)
        bool operator!=(const Shape& other) const {
            return !(*this == other);
        }
    };

    struct Stride
    {
        std::vector<int64_t> strides;
    };

    // Tensor Utility options for smoother API
    struct TensorOptions
    {
        Dtype dtype = Dtype::Float32;
        DeviceIndex device = DeviceIndex(Device::CPU);
        bool requires_grad = false;

        // Builder patterns
        TensorOptions with_dtype(Dtype d) const
        {
            TensorOptions opts = *this;
            opts.dtype = d;
            return opts;
        }
        TensorOptions with_device(DeviceIndex d) const
        {
            TensorOptions opts = *this;
            opts.device = d;
            return opts;
        }

        TensorOptions with_req_grad(bool g) const
        {
            TensorOptions opts = *this;
            opts.requires_grad = g;
            return opts;
        }
    };

    // ########################################################################
    // Class Defintions
    // ########################################################################

    class Tensor
    {
        public:
        //#######################################################
        // Constructor
        //#######################################################

        Tensor(Shape shape, Dtype dtype,
            DeviceIndex device = DeviceIndex(Device::CPU),
            bool requires_grad = false);

        // Constructor with options
        Tensor(Shape shape, TensorOptions opts);

        //✨✨✨
        Tensor(Shape shape, bool requires_grad)
        : Tensor(shape, Dtype::Float32, DeviceIndex(Device::CPU), requires_grad) {}

        Tensor() = default;

        //#######################################################
        // Metadata accessors
        //#######################################################

        const Shape& shape() const { return shape_; };
        const Stride& stride() const { return stride_; };


        Dtype dtype() const { return dtype_; }
        DeviceIndex device() const { return device_; };
        bool requires_grad() const { return  requires_grad_; };
        static size_t dtype_size(Dtype d);
        int64_t ndim() const { return shape_.dims.size(); }

        // ######################################################
        // Data Accessors
        //#######################################################

        void* data() { return data_ptr_.get(); }
        const void* data() const { return data_ptr_.get(); }

        void* grad() { return grad_ptr_.get(); }
        const void* grad() const { return grad_ptr_.get(); }

        // ✨✨✨
        void reset() {
            data_ptr_.reset(); // This is the key line!
            shape_.dims.clear();
            stride_.strides.clear();
            data_size_ = 0;
            storage_offset_ = 0;
        }


        template<typename T>
        T* data()
        {
            return reinterpret_cast<T*>(data_ptr_.get() + storage_offset_);
        }

        template<typename T>
        const T* data() const
        {
            return reinterpret_cast<const T*>(data_ptr_.get() + storage_offset_);
        }

        // ######################################################
        // Static Conditional Operator - Compiler
        //#######################################################
        // In the public section or as standalone function
        template<typename Func1, typename Func2, typename... Args>
        static Tensor cond(bool pred, Func1 true_fn, Func2 false_fn, Args&&... operands) {
            if (pred) {
                return true_fn(std::forward<Args>(operands)...);
            } else {
                return false_fn(std::forward<Args>(operands)...);
            }
        }

        // Element-wise where operation - must be in header (inline or template)
        static Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);
        
        // Overload for scalar input
        static Tensor where(const Tensor& condition, float input_scalar, const Tensor& other);
        
        // Overload for scalar other
        static Tensor where(const Tensor& condition, const Tensor& input, float other_scalar);
        
        // Overload for both scalars
        static Tensor where(const Tensor& condition, float input_scalar, float other_scalar);
        
        // Single argument version (returns indices where condition is true)
        static std::vector<Tensor> where(const Tensor& condition);

        // ######################################################
        // Device Metadata
        //#######################################################
        Tensor to(DeviceIndex device) const;
        Tensor to_cpu() const;
        Tensor to_cuda(int device_index = 0) const;
        bool is_cpu() const;
        bool is_cuda() const;

        Tensor to_bool() const;
        //#######################################################
        // Memory Info
        //#######################################################

        size_t nbytes() const;
        size_t grad_nbytes() const;
        size_t numel() const;
        size_t allocated_bytes() const { return data_size_; }
        size_t grad_allocated_bytes() const { return data_size_; }
        bool owns_data() const { return owns_data_; }
        bool owns_grad() const { return owns_grad_; }
        bool is_contiguous() const;
        Tensor contiguous() const;

        //#######################################################
        // Data Manipulation
        //#######################################################

        template <typename T>
        void set_data(const T* source_data, size_t count);

        template<typename T>
        void set_data(const std::vector<T>& source_data);

        template <typename T>
        void set_data(std::initializer_list<T> values);

        template <typename T>
        void fill(T value);

        //######################################################
        // Factory Functions
        //######################################################

        static Tensor zeros(Shape shape, TensorOptions opts = {});
        static Tensor ones(Shape shape, TensorOptions opts = {});
        static Tensor full(Shape shape, TensorOptions, float val);
        static Tensor rand(Shape shape, TensorOptions opts);
        static Tensor randn(Shape shape, TensorOptions opts);

        //#######################################################
        // View Operations
        //#######################################################
        Tensor view(Shape new_shape) const;
        Tensor reshape(Shape new_shape) const;
        Tensor transpose(int dim0, int dim1) const;
        Tensor t() const;
        Tensor flatten(int start_dim = 0, int end_dim = -1) const;
        Tensor unflatten(int dim, Shape sizes) const;

        //#######################################################
        // View Utilities
        //#######################################################
        size_t storage_offset() const;  // ← MUST EXIST

        //######################################################
        // Utilities
        //######################################################

        void display(std::ostream& os, int prec = 4) const;
        void display() const;
        void display(int prec) const;

        Tensor clone() const;
        Tensor& copy_(const Tensor& src);
        Tensor as_type(Dtype new_dtype) const;

        //######################################################
        // Memory Deletion 
        //######################################################
        void release();
        bool is_valid() const;


        private:
            Shape shape_;
            Stride stride_;
            Dtype dtype_;
            DeviceIndex device_;
            bool requires_grad_;

            // Data Storage using Shared Pointers for Auto Management
            std::shared_ptr<uint8_t[]> data_ptr_;
            std::shared_ptr<uint8_t[]> grad_ptr_;

            // OWNERSHIP FLAGS
            bool owns_data_ = true;
            bool owns_grad_ = true;

            // Size Informations
            size_t storage_offset_ = 0;
            size_t data_size_ = 0;


            Tensor(std::shared_ptr<uint8_t[]> data_ptr,
            Shape shape,
            Stride stride,
            size_t offset,
            Dtype dtype,
            DeviceIndex device,
            bool requires_grad = false);
    };
}

// End of namespace OwnTensor
#include "dtype/DtypeTraits.h"
#include "core/TensorDataManip.h"
#include "core/TensorDispatch.h"