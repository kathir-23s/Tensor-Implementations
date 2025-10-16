#pragma once
#include "Allocator.h"
#include "Device.h"

namespace OwnTensor
{
    class AllocatorRegistry {
    public:
        static Allocator* get_allocator(Device device);
        static Allocator* get_cpu_allocator();
        static Allocator* get_cuda_allocator();
    };
}