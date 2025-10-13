#pragma once

#include <vector>
#include <memory>
#include "device/Device.h"
#include "Types.h"

namespace OwnTensor
{
    // ########################################################################
    // Custom Type Definitions
    // ########################################################################

    // DataTypes and Devices
    enum class Dtype {
        Int16, Int32, Int64,
        Bfloat16, Float16, Float32, Float64
    };



    // Shape and stride
    struct Shape
    {
        std::vector<int64_t> dims;
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

        template <typename T>
        T* data() 
        {
        return reinterpret_cast<T*>(data_ptr_.get());
        }
    
        template <typename T>
        const T* data() const 
        {
            return reinterpret_cast<const T*>(data_ptr_.get());
        }

        // ######################################################
        // Device Metadata
        //#######################################################
        Tensor to(DeviceIndex device) const;
        Tensor to_cpu() const;
        Tensor to_cuda(int device_index = 0) const;
        bool is_cpu() const;
        bool is_cuda() const;


        //#######################################################
        // Memory Info
        //#######################################################

        size_t nbytes() const;
        size_t grad_nbytes() const; 
        size_t numel() const; 
        bool owns_data() const { return owns_data_; }
        bool owns_grad() const { return owns_grad_; }
        bool is_contiguous() const;

        //#######################################################
        // Data Manipulation
        //#######################################################

        template <typename T>
        void set_data(const T* source_data, size_t count);

        template<typename T>
        void set_data(const std::vector<T>& source_data);

        void set_data(std::initializer_list<float> values);
        
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
        
        //######################################################
        // Utilities
        //######################################################
        
        void display(std::ostream& os, int prec) const;
        


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
            size_t data_size_ = 0;
    };  
} 

// End of namespace OwnTensor
#include "DtypeTraits.h"
#include "TensorDataManip.h"
#include "TensorDispatch.h"