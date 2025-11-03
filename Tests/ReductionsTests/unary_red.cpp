#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;

#include <cmath> // for std::nan


int main()
{
    Tensor t_cpu({{3,3}}, Dtype::Float32, DeviceIndex(Device::CPU));
    Tensor t_cpu2({{3,3}}, Dtype::Float32, DeviceIndex(Device::CPU));

    std::vector<float> data1 = {1, 2, 3, nanf(""), 0, 600, -700, 800, 900};

    t_cpu.set_data(data1);
    std::cout << "Original CPU Data: ";
    t_cpu.display(std::cout,6);

    // Tensor t_m({{3,3}}, Dtype::Float16, DeviceIndex(Device::CUDA));
    // t_m.fill(float16_t(223.0f));
    // std::cout << "\n OPERATIONS ON CPU" << std::endl;
    // Tensor t1 = square(t_cpu); //t1 is a cpu tensor
    // t1.display(std::cout,6);

    // square_(t_cpu); //inplace square on cpu tensor
    // std::cout << "\nIn-place square on CPU Tensor: ";
    // t_cpu.display(std::cout,6);
    
    Tensor t1 = reduce_mean(t_cpu); //t1 is a cpu tensor
    t1.display(std::cout,6);

    t1 = reduce_sum(t_cpu);
    t1.display(std::cout,6);

    t1 = reduce_max(t_cpu);
    t1.display(std::cout,6);

    t1= reduce_min(t_cpu);
    t1.display(std::cout,6);

    t1= reduce_nanmax(t_cpu);
    t1.display(std::cout,6);

    t1= reduce_nanmean(t_cpu);
    t1.display(std::cout,6);

    t1= reduce_product(t_cpu);
    t1.display(std::cout,6);

    t1= reduce_nanargmax(t_cpu);
    t1.display(std::cout,6);    

    t1= reduce_argmax(t_cpu);
    t1.display(std::cout,6);    

    t1= reduce_nanargmin(t_cpu);
    t1.display(std::cout,6);    

    t1= reduce_argmin(t_cpu);
    t1.display(std::cout,6);

    t1= reduce_nansum(t_cpu);
    t1.display(std::cout,6);

    t1=reduce_nanmin(t_cpu);
    t1.display(std::cout,6);

    t1=reduce_nanproduct(t_cpu);
    t1.display(std::cout,6);

#ifdef WITH_CUDA
    // Move to GPU
    Tensor t_gpu = t_cpu.to(DeviceIndex(Device::CUDA));  //t_gpu is a gpu tensor
    std::cout << "\n OPERATIONS ON GPU" << std::endl;
    // Tensor t1 = reduce_nanmean(t_gpu); //t1 is a gpu tensor
    // Tensor t2_cpu = t1.to(DeviceIndex(Device::CPU)); // Move back to CPU , t2_cpu is a cpu tensor
    // t2_cpu.display(std::cout,6);

    // Tensor t2 = square(t_gpu); //t2 is a cpu tensor
    // Tensor t2_cpu = t2.to(DeviceIndex(Device::CPU));
    // t2_cpu.display(std::cout,6);

    // square_(t_gpu); //inplace square on gpu tensor
    // t2_cpu = t_gpu.to(DeviceIndex(Device::CPU));
    // std::cout << "\nIn-place square on GPU Tensor: ";
     Tensor t2 = reduce_sum(t_gpu);
    Tensor t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_max(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_min(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_nanmax(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_nanmean(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_product(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_nanargmax(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);    

    t2 = reduce_argmax(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);    

    t2 = reduce_nanargmin(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2 = reduce_argmin(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);
    t1.display(std::cout,6);

    t2= reduce_nansum(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2=reduce_nanmin(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);

    t2=reduce_nanproduct(t_gpu);
    t2_cpu = t2.to(DeviceIndex(Device::CPU));
    t2_cpu.display(std::cout,6);
#endif

//     //INPLACE OPERATION TEST
//     pow_(t_cpu,3); //inplace pow on cpu tensor
//     std::cout << "\nIn-place pow on CPU Tensor: ";
//     t_cpu.display(std::cout,6);

// #ifdef WITH_CUDA
//     pow_(t_gpu,3);
//     std::cout << "\nIn-place pow on GPU operation: ";
//     t1= t_gpu.to(DeviceIndex(Device::CPU));
//     t1.display(std::cout,6);
// #endif

    return 0;
}