#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

// --- REQUIRED CORE FRAMEWORK INCLUDES ---
#include "Tensor.h"     
#include "Types.h"      
#include "Reduction.h" 

// CRITICAL FIX: Include the file that contains the implementation
// of templated functions like set_data and fill.
#include "TensorDataManip.h" 

namespace OwnTensor {

// Utility to print results, relying on the 'display' definition in src/TensorUtils.cpp
void print_reduction_result(const std::string& op_name, const Tensor& result, const std::vector<int64_t>& axes) {
    std::cout << "\n>>> " << op_name << " Result (Axes: ";
    if (axes.empty()) {
        std::cout << "ALL";
    } else {
        for(size_t i = 0; i < axes.size(); ++i) {
            std::cout << axes[i] << (i < axes.size() - 1 ? ", " : "");
        }
    }
    std::cout << ")\n";
    // This calls the display function defined in src/TensorUtils.cpp
    result.display(std::cout , 6); 
}

} // end namespace OwnTensor

// ----------------------------------------------------------------
// MAIN TEST FUNCTION
// ----------------------------------------------------------------

int main() {
    // This line is CRITICAL: It brings all library names (Tensor, Dtype, reduce_max) into scope.
    using namespace OwnTensor;
    
    std::cout << "Starting Reduction Max Test...\n";

    // 1. Create a 2x3 Tensor with float data (Dtype::Float32)
    Tensor T(Shape{{2, 3}}, Dtype::Float16); 
    
    // 2. Set data: [[ 1.0, 5.0, 3.0 ], [ 4.0, 9.0, 6.0 ]]
    // Note: set_data is a templated method defined in TensorDataManip.h
   // T.set_data({1.0f, 5.0f, 3.0f, 4.0f, 9.0f, 6.0f});
T.set_data({float16_t(1.0f), float16_t(5.0f), float16_t(3.0f), float16_t(4.0f), float16_t(9.0f), float16_t(6.0f)});

    std::cout << "\n==================================================\n";
    std::cout << "Input Tensor (2x3): \n";
    T.display(std::cout,6); // Requires display() to be defined in src/TensorUtils.cpp
    std::cout << "==================================================\n";

    // TEST 1: Full Reduction (Scalar Output) - Calling with no arguments
    try {
        Tensor max_full = reduce_max(T); 
        print_reduction_result("reduce_max(T) [Full Reduction]", max_full, {});
    } catch (const std::exception& e) {
        std::cout << "\n!!! EXCEPTION CAUGHT for Full Reduction: " << e.what() << " !!!\n";
    }

    // TEST 2: Axis Reduction (Axis 1) - Reducing columns, KeepDim=False
    // Expected output shape: 2 (Max of each row)
    try {
        std::vector<int64_t> axes1 = {1};
        Tensor max_axis1 = reduce_max(T, axes1); 
        print_reduction_result("reduce_max(T, {1}) [Axis 1]", max_axis1, axes1);
    } catch (const std::exception& e) {
        std::cout << "\n!!! EXCEPTION CAUGHT for Axis 1 Reduction: " << e.what() << " !!!\n";
    }
    
    // TEST 3: Axis Reduction (Axis 0) - Reducing rows, KeepDim=True
    // Expected output shape: 1x3 (Max of each column)
    try {
        std::vector<int64_t> axes0 = {0};
        Tensor max_axis0_keepdim = reduce_max(T, axes0, true); 
        print_reduction_result("reduce_max(T, {0}, true) [Axis 0, KeepDim=True]", max_axis0_keepdim, axes0);
    } catch (const std::exception& e) {
        std::cout << "\n!!! EXCEPTION CAUGHT for Axis 0 Reduction: " << e.what() << " !!!\n";
    }


    std::cout << "\nReduction Max Test finished.\n";
    return 0;
}
