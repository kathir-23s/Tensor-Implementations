//DeviceCore.cpp
#include "device/Device.h"
#include "device/DeviceCore.h"
#include <driver_types.h>//✨✨✨
#ifdef WITH_CUDA
// #include "/usr/include/cuda_runtime.h"
#include <cuda_runtime.h>
#endif

namespace OwnTensor{
    namespace device 
    {
        bool cuda_available() 
        {
            #ifdef WITH_CUDA
                int count;
                return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
            #else
                return false;
            #endif
        }
        
        int cuda_device_count() 
        {
            #ifdef WITH_CUDA
                int count;
                if (cudaGetDeviceCount(&count) == cudaSuccess)
                {
                    return count;
                }
            #endif
                return 0;
        }

        int get_current_cuda_device()
        {
            #ifdef WITH_CUDA
                int device;
                cudaGetDevice(&device);
                return device;
            #else
                return -1;
            #endif
        }

    }
}
//✨✨✨
namespace OwnTensor::cuda {
    static thread_local cudaStream_t g_current_stream = 0; // Default is stream 0

    void setCurrentStream(cudaStream_t stream) {
        g_current_stream = stream;
    }
    cudaStream_t getCurrentStream() {
        return g_current_stream;
    }
}