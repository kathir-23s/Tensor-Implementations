#pragma once
#include "device/Device.h"
#include <memory>

namespace OwnTensor
{
    class Allocator;

    namespace device 
    {
        void copy_memory(void* dst, Device dst_device,
                            const void* src, Device src_device, size_t bytes);
    }
}