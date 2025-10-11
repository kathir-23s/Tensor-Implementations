#include <iostream>
#include <vector>
#include <stdexcept>
#include "../include/Tensor.h"
#include "../include/tesnor_unaryops.hpp"
#include "../include/UnaryDispatcher.hpp"

// =========================================================================
// TEST UTILITY: Tensor Creation
// =========================================================================

/**
 * @brief Creates a Tensor and initializes it with data.
 */
Tensor create_float_tensor(size_t size) {
    // 1. Define Shape and Options
    Shape s{ { (int64_t)size } };
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CPU));

    // 2. Create Tensor (Allocates Memory)
    Tensor t(s, opts); 
    
    // 3. Set Dummy Data (Uses your set_data from Tensor.h)
    std::vector<float> data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    t.set_data(data); 
    
    return t;
}

Tensor create_int_tensor(size_t size) {
    // 1. Define Shape and Options
    Shape s{ { (int64_t)size } };
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Int32).with_device(DeviceIndex(Device::CPU));

    // 2. Create Tensor (Allocates Memory)
    Tensor t(s, opts); 
    
    // 3. Set Dummy Data (Uses your set_data from Tensor.h)
    std::vector<int32_t> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    t.set_data(data); 
    
    return t;
}


// =========================================================================
// SIMPLE EXP TEST
// =========================================================================

void test_exp_return_and_properties() {
    std::cout << "\n--- Test: exp() Return Type and Properties ---\n";
    
    size_t size = 10;
    Tensor input_tensor = create_float_tensor(size);
    //Tensor input_tensor = create_int_tensor(size);
    const void* input_ptr = input_tensor.data();
    
    try {
        // 1. CALL THE HIGH-LEVEL API
        Tensor output_tensor = exp(input_tensor);

        // 2. ASSERTION: Is it a Tensor object? (Checked by compilation/return type)
        std::cout << "SUCCESS: Function returned a Tensor object.\n";

        // 3. ASSERTION: Does it have the correct size?
        if (output_tensor.numel() == input_tensor.numel()) {
            std::cout << "SUCCESS: Output Tensor has the correct size (" << size << ").\n";
        } else {
            std::cerr << "FAIL: Output size is incorrect.\n";
            return;
        }

        // 4. ASSERTION: Does it use a new memory buffer (Out-of-Place check)?
        if (output_tensor.data() != input_ptr) {
            std::cout << "SUCCESS: Output Tensor uses a new memory buffer (Verified Out-of-Place).\n";
        } else {
            std::cerr << "FAIL: Output Tensor shares memory with Input Tensor (Violates Out-of-Place).\n";
        }

        // 5. ASSERTION: Does it have the expected Dtype (Float32 for Float32 input)?
        if (output_tensor.dtype() == Dtype::Float32) {
            std::cout << "SUCCESS: Output Tensor has the expected Dtype (" << dtype_to_string(output_tensor.dtype()) << ").\n";
        } else {
            std::cerr << "FAIL: Output Tensor Dtype is incorrect.\n";
        }

        output_tensor.display(std::cout);

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: Test failed due to unexpected exception during dispatch: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "FATAL ERROR: Test failed due to unknown exception.\n";
    }
}

void test_exp_inline_and_properties() {
    std::cout << "\n--- Test: exp_() Return Type and Properties ---\n";
    
    size_t size = 10;
    Tensor input_tensor = create_float_tensor(size);
    //Tensor input_tensor = create_int_tensor(size);
    const void* input_ptr = input_tensor.data(); 
    
    try {
        // 1. CALL THE HIGH-LEVEL API
        exp_(input_tensor);

        // 2. ASSERTION: Is it a Tensor object? (Checked by compilation/return type)
        std::cout << "SUCCESS: Function did not return a Tensor object.\n";

        // 3. ASSERTION: Does it use a new memory buffer (In-Place check)?
        if (input_tensor.data() == input_ptr) {
            std::cout << "SUCCESS: Output Tensor shares memory with Input Tensor (Verified In-Place)\n";
        } else {
            std::cerr << "FAIL: Output Tensor uses a new memory buffer (Violates In-Place)..\n";
        }

        // 4. ASSERTION: Does it have the expected Dtype (Float32 for Float32 input)?
        if (input_tensor.dtype() == Dtype::Float32) {
            std::cout << "SUCCESS: Tensor has the expected Dtype (" << dtype_to_string(input_tensor.dtype()) << ").\n";
        } else {
            std::cerr << "FAIL: Tensor Dtype is incorrect.\n";
        }

        input_tensor.display(std::cout);

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: Test failed due to unexpected exception during dispatch: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "FATAL ERROR: Test failed due to unknown exception.\n";
    }
}

int main() {
    test_exp_return_and_properties();
    test_exp_inline_and_properties();
    std::cout << "\nSimple Test Complete.\n";
    return 0;
}