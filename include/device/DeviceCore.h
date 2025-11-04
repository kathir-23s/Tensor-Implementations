#pragma once
#include "device/Device.h"
#include <driver_types.h> 
namespace OwnTensor
{
    namespace device {
        bool cuda_available();
        int cuda_device_count();
        // void set_cuda_device(int device_index);
        int get_current_cuda_device();
    }
}
//✨✨✨
namespace OwnTensor::cuda {
    #ifdef WITH_CUDA
        void setCurrentStream(cudaStream_t stream);
        cudaStream_t getCurrentStream();
    #endif
}