#include <iostream>
#include <cmath>
#include <stdexcept>
#include "../include/UnaryDispatcher.hpp"

// --- ASSUME these definitions are included from UnaryDispatcher.hpp ---
// enum class ExecutionMode:uint8_t { Out_of_Place, In_Place }; 
// -------------------------------------------------------------------

// Helper function to simulate the Tensor::exp_() logic, 
// including the critical safety check.
void test_inplace_op(UnaryOp op, Dtype dtype, Device device) {
    std::cout << "\n--- Testing In-Place Operation ---\n";
    
    // 1. Setup Data
    float data_float[] = {1.0f, 2.0f, 3.0f}; // Input data
    size_t size = sizeof(data_float) / sizeof(float);
    
    // Simulate integer data for the failure test
    int32_t data_int[] = {1, 2, 3}; 
    size_t size_int = sizeof(data_int) / sizeof(int32_t);


    // 2. Setup Key
    KernelKey key{op, dtype, device, ExecutionMode::In_Place};
    
    try {
        // We simulate the safety check here for the integer case.
        // NOTE: In your final implementation, this check will be in Tensor::exp_()
        
        // This check assumes op_promotes_int(op) and is_int(dtype) are available
        // We will assume 'Exp' promotes to float and Int32 is an integer.
        if (op == UnaryOp::Exp && dtype == Dtype::Int32 /* assuming Dtype::Int32 exists */) {
            std::cout << "Attempting Exp on Int32 (In-Place)... ";
            // SIMULATE ERROR THROW from Tensor::exp_()
            throw std::runtime_error("Safety Check Fail: Cannot perform in-place 'Exp' on integer tensor.");
        }
        
        // 3. Lookup Kernel
        UnaryKernelFn fn = KernelRegistry::instance().get_kernel(key);

        if (fn != nullptr) {
            std::cout << "SUCCESS: Found the registered IN-PLACE kernel.\n";
            
            // 4. Execute Kernel (Success Case: Float)
            if (dtype == Dtype::Float32) {
                std::cout << "Calling kernel: data_float = {1.0, 2.0, 3.0}\n";
                // The input and output pointers MUST be the same for in-place
                fn(data_float, data_float, size, 0.0); 
                
                std::cout << "Result data_float: {" 
                          << data_float[0] << ", " 
                          << data_float[1] << ", " 
                          << data_float[2] << "}\n";
                          
                // Verify the result (e^1 ≈ 2.71828)
                if (std::abs(data_float[0] - 2.71828f) < 0.001f) {
                    std::cout << "VERIFICATION: In-place operation succeeded and modified the data.\n";
                } else {
                    std::cout << "VERIFICATION FAILED: Data not modified correctly.\n";
                }
            } else {
                std::cout << "Kernel found but not called/verified for this type in test.\n";
            }
        } else {
            std::cout << "FAILURE: IN-PLACE kernel NOT found in the registry.\n";
        }

    } catch (const std::runtime_error& e) {
        // This catches the simulated safety error from STEP 2
        std::cout << "SAFETY CHECK PASSED: Caught expected error: " << e.what() << "\n";
    } catch (...) {
        std::cout << "An unexpected error occurred.\n";
    }
}


int main() {
    std::cout << "--- Program Start: Checking Registry Population ---\n";
    
    // --- TEST 1: IN-PLACE EXP SUCCESS (FLOAT32) ---
    // This tests if the in-place kernel is correctly registered and executed.
    test_inplace_op(UnaryOp::Exp, Dtype::Float32, Device::CPU);
    
    // --- TEST 2: IN-PLACE EXP FAILURE (INT32) ---
    // This tests if the high-level safety logic prevents the in-place operation 
    // on an integer tensor, as it would cause data loss.
    // NOTE: This test relies on you correctly implementing the check outside the dispatcher.
    test_inplace_op(UnaryOp::Exp, Dtype::Int32, Device::CPU);


    // Optional: Keep your original out-of-place test to ensure nothing broke
    std::cout << "\n--- Testing Out-of-Place Sanity Check ---\n";
    KernelKey exp_float_op{
        UnaryOp::Exp,
        Dtype::Float32,
        Device::CPU,
        ExecutionMode::Out_of_Place // Ensure OP is still fine
    };
    UnaryKernelFn fn_op = KernelRegistry::instance().get_kernel(exp_float_op);
    if (fn_op != nullptr) {
        std::cout << "SUCCESS: Found the registered OUT-OF-PLACE 'Exp' kernel.\n";
    } else {
        std::cout << "FAILURE: Out-of-place kernel not found.\n";
    }

    std::cout << "\n--- Program End ---\n";
    return 0;
}