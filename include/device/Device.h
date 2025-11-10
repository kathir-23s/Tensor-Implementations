#pragma once

namespace OwnTensor
{
    enum class Device 
    {
        CPU,
        CUDA
    };

    struct DeviceIndex 
    {
        Device device;
        int index;

        DeviceIndex(Device dev = Device::CPU, int idx = 0) : device(dev), index(idx) {}

        bool is_cpu() const { return device == Device::CPU;}
        bool is_cuda() const { return device == Device::CUDA;}
    };
}

// Tensor(shape, dtype, (DeviceIndex(Device::CUDA), 0))
// Tensor(shape, dtype, Device::CUDA);