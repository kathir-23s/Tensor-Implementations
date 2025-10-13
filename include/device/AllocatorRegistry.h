#pragma once
#include "device/Allocator.h"
#include "device/Device.h"

namespace OwnTensor
{
    class AllocatorRegistry {
    public:
        static Allocator* get_allocator(Device device);
        static Allocator* get_cpu_allocator();
        static Allocator* get_cuda_allocator();
    };
}