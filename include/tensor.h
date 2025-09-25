#pragma once

#include <cuda_stdint.h>
#include <vector>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <cstring>


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
    std::vector<int32_t> dims;
};

struct Stride
{
    std::vector<int32_t> strides;
};


class Tensor 
{
    public:
    // Constructor
        Tensor(Shape shape, Dtype dtype, DeviceIndex device = DeviceIndex(Device::CPU), bool requires_grad = false);

    // Metadata accessors
    std::vector<int32_t> shape() const { return shape_.dims; };
    std::vector<int32_t> stride() const { return stride_.strides; };
    Dtype dtype() const { return dtype_; }
    DeviceIndex device() const { return device_; };
    bool requires_grad() const { return  requires_grad_; };
    bool is_owner() const { return is_owner_; };


    private:
        Shape shape_;
        Stride stride_;
        Dtype dtype_;
        DeviceIndex device_;
        bool requires_grad_;
        bool is_owner_;
};  



