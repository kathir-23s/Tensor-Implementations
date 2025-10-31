#include <iostream>
#include <vector>
#include <string>
#include "TensorLib.h" // Your main library header

using namespace OwnTensor;

// Helper function to convert Dtype enum to a printable string.
// This is just for making the test output readable.
std::string dtype_to_string(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float32: return "Float32";
        case Dtype::Int32:   return "Int32";
        case Dtype::Float64: return "Float64";
        // Add other Dtype cases as you need them for testing.
        default:             return "Unknown Dtype";
    }
}

int main() {
    std::cout << "--- Testing Tensor.as_type() ---" << std::endl;

    // 1. Create a source tensor with Float32 data.
    Tensor tensor_a(Shape{{2, 3}}, TensorOptions{Dtype::Float32, Device::CPU});
    std::vector<float> data = {1.1f, -2.2f, 3.9f, 4.0f, 5.5f, -6.7f};
    tensor_a.set_data(data);

    std::cout << "\nOriginal Tensor (A):" << std::endl;
    // Use the helper function for printing the dtype
    std::cout << "Dtype: " << dtype_to_string(tensor_a.dtype()) << std::endl;
    tensor_a.display(std::cout, 4);

    // 2. Use the new as_type() method to convert it to Int32.
    std::cout << "\nConverting Tensor A to Int32..." << std::endl;
    Tensor tensor_b = tensor_a.as_type(Dtype::Int32);

    // 3. Display the new tensor to verify the result.
    // The floating point values should be truncated (e.g., 3.9f becomes 3).
    std::cout << "\nNew Tensor (B):" << std::endl;
    // Use the helper function again for the new tensor's dtype
    std::cout << "Dtype: " << dtype_to_string(tensor_b.dtype()) << std::endl;
    tensor_b.display(std::cout, 4);

    std::cout << "\n--- Test Complete ---" << std::endl;

    return 0;
}