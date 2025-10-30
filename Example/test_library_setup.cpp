#include <iostream>
#include "TensorLib.h"

int main()
{
    std::cout << "--- Code is Running ---" << std::endl;
    std::cout << "Testing TensorLib Integration" <<std::endl;

    try 
    {
        OwnTensor::Tensor my_tensor(
            OwnTensor::Shape{{2,2}}, 
            OwnTensor::TensorOptions{
                OwnTensor::Dtype::Float32,
                OwnTensor::Device::CPU,
                false}
        );

        my_tensor.fill(3.14f);

        std::cout << "Successfully created and displayed a Tensor:" << std::endl;
        my_tensor.display(std::cout, 4);
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n---  TensorLib Integration Test Successful ---" << std::endl;
    return 0;
}