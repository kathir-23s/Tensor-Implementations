#pragma once

#include <vector>
#include <memory>


// ########################################################################
// Custom Type Definitions
// ########################################################################

// DataTypes and Devices
enum class Dtype {
    Int16, Int32, Int64,
    Bfloat16, Float16, Float32, Float64
};

enum class Device {
    CPU,
    CUDA
};

struct DeviceIndex {
    Device device;
    int index;

    DeviceIndex(Device dev, int idx = 0) : device(dev), index(idx) {}
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
    // Constructor
    Tensor(Shape shape, Dtype dtype, 
        DeviceIndex device = DeviceIndex(Device::CPU), 
        bool requires_grad = false);

    // Constructor with options
    Tensor(Shape shape, TensorOptions opts);

    // Metadata accessors
    std::vector<int64_t> shape() const { return shape_.dims; };
    std::vector<int64_t> stride() const { return stride_.strides; };
    Dtype dtype() const { return dtype_; }
    DeviceIndex device() const { return device_; };
    bool requires_grad() const { return  requires_grad_; };
    static size_t dtype_size(Dtype d);

    // Data Accessors
    void* data() { return data_ptr_.get(); }
    const void* data() const { return data_ptr_.get(); }

    void* grad() { return grad_ptr_.get(); }
    const void* grad() const { return grad_ptr_.get(); }
    
    // Memory Info
    size_t nbytes() const;
    size_t grad_nbytes() const; 
    size_t numel() const; 
    bool owns_data() const { return owns_data_; }
    bool owns_grad() const { return owns_grad_; }


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



