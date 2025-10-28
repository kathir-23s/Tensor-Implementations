#include "TensorLib.h"
// #include "../include/Debug.h"

#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void test_tensor_stride_calc_1d() 
{
    cout << "\n=== test_tensor_stride_calc_1d ===\n" << endl;

    cout << "CPU: Testing valid stride checks...\n" << endl;
    {
        Tensor t1(Shape{{10}}, Dtype::Float32, DeviceIndex(Device::CPU));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 1);
        assert(strides[0] == 1);

        cout << "✓ 1D tensor [10]: numel=" << t1.numel() 
                  << ", stride=" << strides[0] << endl;
    }

    {
        // cout << "CPU: Testing valid stride checks...\n" << endl;
        Tensor t2(Shape{{1}}, Dtype::Float32, DeviceIndex(Device::CPU));
    
        const auto& strides = t2.stride().strides;
        assert(strides.size() == 1);
        assert(strides[0] == 1);

        cout << "✓ 1D tensor [1]: numel=" << t2.numel() 
                  << ", stride=" << strides[0] << endl;
    }

    {
        // cout << "CPU: Testing valid stride checks...\n" << endl;
        Tensor t3(Shape{{1000}}, Dtype::Float32, DeviceIndex(Device::CPU));
        
        const auto& strides = t3.stride().strides;
        assert(strides.size() == 1);
        assert(strides[0] == 1);

        cout << "✓ 1D tensor [1000]: numel=" << t3.numel() 
                  << ", stride=" << strides[0] << endl;
    }

    cout << "GPU: Testing valid stride checks...\n" << endl;
    {
        Tensor t1(Shape{{10}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 1);
        assert(strides[0] == 1);

        cout << "✓ 1D tensor [10]: numel=" << t1.numel() 
                  << ", stride=" << strides[0] << endl;
    }

    {
        // cout << "CPU: Testing valid stride checks...\n" << endl;
        Tensor t2(Shape{{1}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    
        const auto& strides = t2.stride().strides;
        assert(strides.size() == 1);
        assert(strides[0] == 1);

        cout << "✓ 1D tensor [1]: numel=" << t2.numel() 
                  << ", stride=" << strides[0] << endl;
    }

    {
        // cout << "CPU: Testing valid stride checks...\n" << endl;
        Tensor t3(Shape{{100000000}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        const auto& strides = t3.stride().strides;
        assert(strides.size() == 1);
        assert(strides[0] == 1);

        cout << "✓ 1D tensor [1000000000]: numel=" << t3.numel() 
                  << ", stride=" << strides[0] << endl;
                  
    }

}

void test_tensor_stride_calc_2d() 
{
    cout << "\n=== test_tensor_stride_calc_2d ===\n" << endl;

    cout << "CPU: Testing 2D tensor strides...\n" << endl;
    {
        Tensor t1(Shape{{3, 4}}, Dtype::Float32, DeviceIndex(Device::CPU));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 2);
        assert(strides[0] == 4);  // Stride for dimension 0
        assert(strides[1] == 1);  // Stride for dimension 1
        assert(t1.numel() == 12);

        cout << "✓ 2D tensor [3, 4]: numel=" << t1.numel() 
                  << ", strides=[" << strides[0] << ", " << strides[1] << "]" << endl;
    }

    cout << "GPU: Testing 2D tensor strides...\n" << endl;
    {
        Tensor t1(Shape{{5, 6}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 2);
        assert(strides[0] == 6);  // Stride for dimension 0
        assert(strides[1] == 1);  // Stride for dimension 1
        assert(t1.numel() == 30);

        cout << "✓ 2D tensor [5, 6]: numel=" << t1.numel() 
                  << ", strides=[" << strides[0] << ", " << strides[1] << "]" << endl;
    }
}

void test_tensor_stride_calc_3d() 
{
    cout << "\n=== test_tensor_stride_calc_3d ===\n" << endl;

    cout << "CPU: Testing 3D tensor strides...\n" << endl;
    {
        Tensor t1(Shape{{2, 3, 4}}, Dtype::Float32, DeviceIndex(Device::CPU));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 3);
        assert(strides[0] == 12); // Stride for dimension 0: 3 * 4
        assert(strides[1] == 4);  // Stride for dimension 1: 4
        assert(strides[2] == 1);  // Stride for dimension 2: 1
        assert(t1.numel() == 24);

        cout << "✓ 3D tensor [2, 3, 4]: numel=" << t1.numel() 
                  << ", strides=[" << strides[0] << ", " << strides[1] << ", " << strides[2] << "]" << endl;
    }

    cout << "GPU: Testing 3D tensor strides...\n" << endl;
    {
        Tensor t1(Shape{{3, 4, 5}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 3);
        assert(strides[0] == 20); // Stride for dimension 0: 4 * 5
        assert(strides[1] == 5);  // Stride for dimension 1: 5
        assert(strides[2] == 1);  // Stride for dimension 2: 1
        assert(t1.numel() == 60);

        cout << "✓ 3D tensor [3, 4, 5]: numel=" << t1.numel() 
                  << ", strides=[" << strides[0] << ", " << strides[1] << ", " << strides[2] << "]" << endl;
    }
}

void test_tensor_stride_calc_4d() 
{
    cout << "\n=== test_tensor_stride_calc_4d ===\n" << endl;

    cout << "CPU: Testing 4D tensor strides...\n" << endl;
    {
        Tensor t1(Shape{{2, 3, 4, 5}}, Dtype::Float32, DeviceIndex(Device::CPU));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 4);
        assert(strides[0] == 60); // Stride for dimension 0: 3 * 4 * 5
        assert(strides[1] == 20); // Stride for dimension 1: 4 * 5
        assert(strides[2] == 5);  // Stride for dimension 2: 5
        assert(strides[3] == 1);  // Stride for dimension 3: 1
        assert(t1.numel() == 120);

        cout << "✓ 4D tensor [2, 3, 4, 5]: numel=" << t1.numel() 
                  << ", strides=[" << strides[0] << ", " << strides[1] << ", " 
                  << strides[2] << ", " << strides[3] << "]" << endl;
    }

    cout << "GPU: Testing 4D tensor strides...\n" << endl;
    {
        Tensor t1(Shape{{4, 5, 6, 7}}, Dtype::Float32, DeviceIndex(Device::CUDA));
        
        const auto& strides = t1.stride().strides;
        assert(strides.size() == 4);
        assert(strides[0] == 210); // Stride for dimension 0: 5 * 6 * 7
        assert(strides[1] == 42);  // Stride for dimension 1: 6 * 7
        assert(strides[2] == 7);   // Stride for dimension 2: 7
        assert(strides[3] == 1);   // Stride for dimension 3: 1
        assert(t1.numel() == 840);

        cout << "✓ 4D tensor [4, 5, 6, 7]: numel=" << t1.numel() 
                  << ", strides=[" << strides[0] << ", " << strides[1] << ", " 
                  << strides[2] << ", " << strides[3] << "]" << endl;
    }
}

int main()
{
    test_tensor_stride_calc_1d();
    test_tensor_stride_calc_2d();
    test_tensor_stride_calc_3d();
    test_tensor_stride_calc_4d();

    return 0;
}