#include <iostream>
#include "TensorLib.h"
using namespace OwnTensor;

int main() {
    Shape sh{{2, 3}};
    TensorOptions opts(Dtype::Int32, DeviceIndex(Device::CPU,0),true);
    TensorOptions optsf(Dtype::Int32, DeviceIndex(Device::CUDA,0));
    Tensor a(sh, opts);
    a.fill(int(5));
    auto* grad_ptr = static_cast<int32_t*>(a.grad());
    for (size_t i=0; i<a.numel();i++)
    {
        grad_ptr[i] = i;
    }
//    Tensor b = a.to_cpu() ;
    a.display(std::cout, 6 );
    return 0;
}