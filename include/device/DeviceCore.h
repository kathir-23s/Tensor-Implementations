#pragma once
#include "device/Device.h"

namespace OwnTensor
{
    namespace device {
        bool cuda_available();
        int cuda_device_count();
        void set_cuda_device(int device_index);
        int get_current_cuda_device();
    }
}