#include "TensorLib.h"
#include <cuda_bf16.h>
#include <iostream>

using namespace OwnTensor;
using namespace std;

int main() {
    try {
        
        {          
            Tensor A = Tensor::rand(Shape{{4, 6}}, 
                TensorOptions{Dtype::Bfloat16, DeviceIndex(Device::CUDA)});            
            Tensor B = Tensor::rand(Shape{{4, 1}}, TensorOptions{Dtype::Bfloat16, DeviceIndex(Device::CUDA)});
            // Debug: Print shapes
            std::cout << "A shape: " << A.shape().dims[0] << "x" << A.shape().dims[1] << std::endl;
            std::cout << "B shape: " << B.shape().dims[0] << "x" << B.shape().dims[1] << std::endl;
            
            // A.fill<__nv_bfloat16>(__nv_bfloat16(3.1));
            // B.fill<__nv_bfloat16>(__nv_bfloat16(1.025));
            
            std::cout << "\nBefore Operations...." << std::endl;
            A.display(cout, 8);
            cout << "\n";
            B.display(cout, 8);
            cout << "\n";
            std::cout << "=== Test 1: CPU Addition ===" << std::endl;
            Tensor C = A + B;
            A += B;
            
            cout << "Outplace Tensor Operation: Output (New Tensor C) " << endl;
            cout << "\n";
            C.display(cout, 8);
            
            cout << "Inplace Tensor Operation: Output (Tensor A) " << endl;
            cout << "\n";
            A.display(cout, 8);
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}