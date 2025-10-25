#include "TensorLib.h"
#include "dtype/DtypeTraits.h"
#include <iostream>
#include <type_traits>
#include <cstdint>
#include <cassert>
#include <vector>

using namespace std;
using namespace OwnTensor;

void test_dtype_traits()
{
    {
        cout << "=== Testing Dtype Sizes and Correspondence ===" << endl;

        vector<pair<Dtype, int64_t>> test_cases = {
            {Dtype::Float32, sizeof(float)},
            {Dtype::Float64, sizeof(double)},
            {Dtype::Int16, sizeof(int16_t)},
            {Dtype::Int32, sizeof(int32_t)},
            {Dtype::Int64, sizeof(int64_t)}
        };

        cout << "\nCPU Side Checking for existing types" << endl;
        for (const auto& [dtype, expected] : test_cases) {
            Tensor t(Shape{{10,10}}, dtype, DeviceIndex(Device::CPU), false);
            assert(t.dtype_size(t.dtype()) == expected);
            Dtype dtypeT = t.dtype();
            cout << "✓ Dtype Label " << get_dtype_name(dtypeT) << ": size=" << (t.dtype_size(dtypeT)) << endl;
            cout << "✓ Dtype Flag Test: Is integer Type: " << is_int(dtypeT) << " | Is Float Type: " << is_float(dtypeT) << "\n" << endl;
            // cout << "✓ Test - C++ Dtype to Dtype Structure: " << type_to_dtype()
        }

        
        cout << "\nCUDA Side Checking for existing types" << endl;
        for (const auto& [dtype, expected] : test_cases) {
            Tensor t(Shape{{10,10}}, dtype, DeviceIndex(Device::CUDA), false);
            assert(t.dtype_size(t.dtype()) == expected);
            Dtype dtypeT = t.dtype();
            cout << "✓ Dtype Label " << get_dtype_name(dtypeT) << ": size=" << (t.dtype_size(t.dtype())) << endl;
            cout << "✓ Dtype Flag Test: Is integer Type: " << is_int(dtypeT) << " | Is Float Type: " << is_float(dtypeT) << "\n" << endl;
        }
    }
}

void test_type_to_dtype_mapping() {
    auto check = [](auto value, Dtype expected, const char* type_name) {
        Dtype got = type_to_dtype<decltype(value)>();
        std::cout << "Testing type: " << type_name
                  << " | Expected: " << static_cast<int>(expected)
                  << " | Got: " << static_cast<int>(got)
                  << " | Match: " << (got == expected ? "YES" : "NO")
                  << std::endl;
        assert(got == expected);
    };

    // Only use exact types your type_to_dtype() supports
    check(int16_t{}, Dtype::Int16, "int16_t");
    check(int32_t{}, Dtype::Int32, "int32_t");
    check(int64_t{}, Dtype::Int64, "int64_t");
    check(float16_t{}, Dtype::Float16, "float16");
    check(bfloat16_t{}, Dtype::Bfloat16, "bfloat16");
    check(float{}, Dtype::Float32, "float");
    check(double{}, Dtype::Float64, "double");

    std::cout << "All supported type_to_dtype mappings passed!" << std::endl;
}



int main ()
{
    test_dtype_traits();
    // test_type_to_dtype_mapping();
}
