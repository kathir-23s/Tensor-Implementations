#include "TensorLib.h"
#include "../include/dtype/DtypeTraits.h"
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>
#include <string>

using namespace std;
using namespace OwnTensor;

// Template function instead of generic lambda
template<typename T>
void check_type_mapping(Dtype expected, const char* type_name) {
    Dtype got = type_to_dtype<T>();
    cout << "Testing type: " << type_name
          << " | Expected: " << static_cast<int>(expected)
          << " | Got: " << static_cast<int>(got)
          << " | Match: " << (got == expected ? "YES" : "NO")
          << endl;
    assert(got == expected);
}

void test_type_to_dtype_mapping() {
    cout << "=== Testing type_to_dtype Mapping ===" << endl;
    
    // Test all supported types using template function
    check_type_mapping<int16_t>(Dtype::Int16, "int16_t");
    check_type_mapping<int32_t>(Dtype::Int32, "int32_t");
    check_type_mapping<int64_t>(Dtype::Int64, "int64_t");
    check_type_mapping<float>(Dtype::Float32, "float");
    check_type_mapping<double>(Dtype::Float64, "double");

    // Test built-in types that should map to the same Dtype
    check_type_mapping<short>(Dtype::Int16, "short");
    check_type_mapping<int>(Dtype::Int32, "int");
    check_type_mapping<long>(Dtype::Int64, "long");
    check_type_mapping<long long>(Dtype::Int64, "long long");

    cout << "All type_to_dtype mappings passed!\n" << endl;
}

// Rest of the functions remain the same...
void test_is_same_type_validation() {
    cout << "=== Testing is_same_type Validation ===" << endl;
    
    auto test_case = [](Dtype tensor_dtype, auto value, const char* type_name) {
        bool result = is_same_type<decltype(value)>(tensor_dtype);
        cout << "Type: " << type_name 
             << " | Expected: true"
             << " | Got: " << (result ? "true" : "false")
             << " | " << (result ? "✓ PASS" : "✗ FAIL")
             << endl;
        assert(result == true);
    };

    test_case(Dtype::Int32, int32_t{}, "int32_t");
    test_case(Dtype::Int64, int64_t{}, "int64_t");
    test_case(Dtype::Float32, float{}, "float");
    test_case(Dtype::Float64, double{}, "double");
    test_case(Dtype::Int16, int16_t{}, "int16_t");

    auto test_mismatch = [](Dtype tensor_dtype, auto wrong_value, const char* tensor_type, const char* test_type) {
        bool result = is_same_type<decltype(wrong_value)>(tensor_dtype);
        cout << "Tensor type: " << tensor_type 
             << " | Test type: " << test_type
             << " | Expected: false"
             << " | Got: " << (result ? "true" : "false")
             << " | " << (result ? "✗ FAIL" : "✓ PASS")
             << endl;
        assert(result == false);
    };

    test_mismatch(Dtype::Int32, float{}, "Int32", "float");
    test_mismatch(Dtype::Float64, int32_t{}, "Float64", "int32_t");
    test_mismatch(Dtype::Int16, double{}, "Int16", "double");
    test_mismatch(Dtype::Float32, int64_t{}, "Float32", "int64_t");

    cout << "All is_same_type validation tests passed!\n" << endl;
}

void test_dtype_size_consistency() {
    cout << "=== Testing dtype_size Consistency with dtype_traits ===" << endl;
    
    vector<pair<Dtype, string>> test_cases = {
        {Dtype::Int16, "Int16"},
        {Dtype::Int32, "Int32"},
        {Dtype::Int64, "Int64"},
        {Dtype::Float32, "Float32"},
        {Dtype::Float64, "Float64"}
    };

    auto get_traits_size = [](Dtype dtype) -> size_t {
        switch (dtype) {
            case Dtype::Int16: return dtype_traits<Dtype::Int16>::size;
            case Dtype::Int32: return dtype_traits<Dtype::Int32>::size;
            case Dtype::Int64: return dtype_traits<Dtype::Int64>::size;
            case Dtype::Float32: return dtype_traits<Dtype::Float32>::size;
            case Dtype::Float64: return dtype_traits<Dtype::Float64>::size;
            default: return 0;
        }
    };

    cout << "Testing CPU tensors:" << endl;
    for (const auto& test_case : test_cases) {
        Dtype dtype = test_case.first;
        string type_name = test_case.second;
        
        Tensor t(Shape{{10}}, dtype, DeviceIndex(Device::CPU), false);
        size_t tensor_size = t.dtype_size(t.dtype());
        size_t traits_size = get_traits_size(dtype);
        
        cout << "Dtype: " << type_name 
             << " | Tensor size: " << tensor_size 
             << " | Traits size: " << traits_size
             << " | Match: " << (tensor_size == traits_size ? "✓ YES" : "✗ NO")
             << endl;
        
        assert(tensor_size == traits_size);
    }

    cout << "\nTesting CUDA tensors:" << endl;
    for (const auto& test_case : test_cases) {
        Dtype dtype = test_case.first;
        string type_name = test_case.second;
        
        Tensor t(Shape{{10}}, dtype, DeviceIndex(Device::CUDA), false);
        size_t tensor_size = t.dtype_size(t.dtype());
        size_t traits_size = get_traits_size(dtype);
        
        cout << "Dtype: " << type_name 
             << " | Tensor size: " << tensor_size 
             << " | Traits size: " << traits_size
             << " | Match: " << (tensor_size == traits_size ? "✓ YES" : "✗ NO")
             << endl;
        
        assert(tensor_size == traits_size);
    }

    cout << "All dtype_size consistency tests passed!\n" << endl;
}

int main() {
    cout << "Starting Tensor Type System Tests...\n" << endl;
    
    test_type_to_dtype_mapping();
    test_is_same_type_validation();
    test_dtype_size_consistency();
    
    cout << "All Tensor type system tests passed successfully!" << endl;
    return 0;
}