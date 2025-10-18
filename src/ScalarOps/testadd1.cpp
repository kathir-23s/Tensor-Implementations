#include <iostream>
#include <vector>
#include "Tensor.h"   // Your Tensor class
#include "ScalarOps.h" // The file where you defined operator+ etc. inside namespace OwnTensor

using namespace OwnTensor;

int main() {
    try {
        std::cout << "=== Scalar-Tensor Operator Overload Test ===\n";

        // ---- Test 1: Float32 Tensor ----
        std::cout << "\n[Float32 Tensor Test]\n";
        Tensor t1({{5}}, Dtype::Float32, Device::CPU);
        std::vector<float> data1 = {1.0f, 2.0f, 3.5f, -1.5f, 0.0f};
        t1.set_data(data1);

        float scalar_f = 2.0f;

        Tensor add_f = t1 + scalar_f;
        Tensor add_f2 = scalar_f + t1; // commutative check

        std::cout << "t1 + " << scalar_f << " = ";
        add_f.display(std::cout, 5);
        std::cout << scalar_f << " + t1 = ";
        add_f2.display(std::cout, 5);
        std::cout << "t1 - " << scalar_f << " = ";

        // ---- Test 2: Int32 Tensor ----
        std::cout << "\n[Int32 Tensor Test]\n";
        Tensor t2({{5}}, Dtype::Int32, Device::CPU);
        std::vector<int32_t> data2 = {1, 2, 3, 4, 5};
        t2.set_data(data2);

        int scalar_i = 10;

        Tensor add_i = t2 + scalar_i;
        Tensor add_i2 = scalar_i + t2;

        std::cout << "t2 + " << scalar_i << " = ";
        add_i.display(std::cout, 5);
        std::cout << scalar_i << " + t2 = ";
        add_i2.display(std::cout, 5);
        std::cout << "t2 - " << scalar_i << " = ";

        std::cout << "\n✅ All scalar operations executed successfully!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
    }

    return 0;
}
