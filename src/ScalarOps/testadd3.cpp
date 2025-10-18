#include <iostream>
#include <iomanip>
#include "Tensor.h"
#include "ScalarOps.h"

using namespace OwnTensor;

void test_int_int_promotion() {
    std::cout << "\n=== Testing Integer + Integer Promotion ===" << std::endl;
    
    // Int16 + Int16 -> Int16
    Tensor a16({{3}}, Dtype::Int16);
    int16_t* data_a16 = static_cast<int16_t*>(a16.data());
    data_a16[0] = 1; data_a16[1] = 2; data_a16[2] = 3;
    Tensor result1 = a16 + static_cast<int16_t>(5);
    std::cout << "Int16 + Int16\n";
    result1.display(std::cout, 5);
    
    // Int16 + Int32 -> Int32 (scalar int maps to int64, promotes to int64)
    Tensor result2 = a16 + 5;  // int -> int64 in PyTorch
    std::cout << "\nInt16 + Int32\n";
    result2.display(std::cout, 5);
    
    // Int32 + Int32 -> Int32
    Tensor a32({{3}}, Dtype::Int32);
    int32_t* data_a32 = static_cast<int32_t*>(a32.data());
    data_a32[0] = 10; data_a32[1] = 20; data_a32[2] = 30;
    Tensor result3 = a32 + static_cast<int32_t>(5);
    std::cout << "\nInt32 + Int32\n";
    result3.display(std::cout, 5);
    
    // Int32 + Int64 -> Int64
    Tensor result4 = a32 + static_cast<int64_t>(5);
    std::cout << "\nInt32 + Int64\n";
    result4.display(std::cout, 5);
    
    // Int64 + Int64 -> Int64
    Tensor a64({{3}}, Dtype::Int64);
    int64_t* data_a64 = static_cast<int64_t*>(a64.data());
    data_a64[0] = 100; data_a64[1] = 200; data_a64[2] = 300;
    Tensor result5 = a64 + static_cast<int64_t>(50);
    std::cout << "\nInt64 + Int64\n";
    result5.display(std::cout, 5);
}

void test_int_float_promotion() {
    std::cout << "\n=== Testing Integer + Float Promotion ===" << std::endl;
    
    // Int16 + Float32 -> Float32
    Tensor a16({{3}}, Dtype::Int16);
    int16_t* data_a16 = static_cast<int16_t*>(a16.data());
    data_a16[0] = 1; data_a16[1] = 2; data_a16[2] = 3;
    Tensor result1 = a16 + 2.5f;
    std::cout << "\nInt16 + Float32\n";
    result1.display(std::cout, 5);
    
    // Int32 + Float32 -> Float32
    Tensor a32({{3}}, Dtype::Int32);
    int32_t* data_a32 = static_cast<int32_t*>(a32.data());
    data_a32[0] = 10; data_a32[1] = 20; data_a32[2] = 30;
    Tensor result2 = a32 + 1.5f;
    std::cout << "\nInt32 + Float32\n";
    result2.display(std::cout, 5);
    
    // Int64 + Float32 -> Float32
    Tensor a64({{3}}, Dtype::Int64);
    int64_t* data_a64 = static_cast<int64_t*>(a64.data());
    data_a64[0] = 100; data_a64[1] = 200; data_a64[2] = 300;
    Tensor result3 = a64 + 0.5f;
    std::cout << "\nInt64 + Float32\n";
    result3.display(std::cout, 5);
    
    // Int32 + Float64 -> Float64 (double scalar -> float32 in your impl, but let's test)
    Tensor result4 = a32 + 2.5;  // double -> float32 based on your scalar_to_dtype
    std::cout << "\nInt32 + double(float64)\n";
    result4.display(std::cout, 5);
}

void test_float_float_promotion() {
    std::cout << "\n=== Testing Float + Float Promotion ===" << std::endl;
    
    // Float16 + Float16 -> Float16
    Tensor f16({{3}}, Dtype::Float16);
    float16_t* data_f16 = static_cast<float16_t*>(f16.data());
    data_f16[0] = float16_t(1.0f); 
    data_f16[1] = float16_t(2.0f); 
    data_f16[2] = float16_t(3.0f);
    Tensor result1 = f16 + 0.5f;
    std::cout << "\nFloat16 + Float32\n";
    result1.display(std::cout, 5);
    
    // Bfloat16 + Bfloat16 -> Bfloat16
    Tensor bf16({{3}}, Dtype::Bfloat16);
    bfloat16_t* data_bf16 = static_cast<bfloat16_t*>(bf16.data());
    data_bf16[0] = bfloat16_t(1.0f);
    data_bf16[1] = bfloat16_t(2.0f);
    data_bf16[2] = bfloat16_t(3.0f);
    Tensor result2 = bf16 + 0.5f;
    std::cout << "\nBfloat16 + Float32\n";
    result2.display(std::cout, 5);
    
    // Float32 + Float32 -> Float32
    Tensor f32({{3}}, Dtype::Float32);
    float* data_f32 = static_cast<float*>(f32.data());
    data_f32[0] = 1.1f; data_f32[1] = 2.2f; data_f32[2] = 3.3f;
    Tensor result3 = f32 + 0.5f;
    std::cout << "\nFloat32 + Float32\n";
    result3.display(std::cout, 5);
    
    // Float64 + Float64 -> Float64
    Tensor f64({{3}}, Dtype::Float64);
    double* data_f64 = static_cast<double*>(f64.data());
    data_f64[0] = 1.11; data_f64[1] = 2.22; data_f64[2] = 3.33;
    Tensor result4 = f64 + 0.5f;  // float32 scalar + float64 tensor -> float64
    std::cout << "\nFloat64 + Float32\n";
    result4.display(std::cout, 5);
}

void test_special_cases() {
    std::cout << "\n=== Testing Special Promotion Cases ===" << std::endl;
    
    // Bfloat16 + Float16 -> Float32 (special PyTorch rule)
    // Since we can't easily create this in scalar ops, this would need tensor+tensor ops
    // For now, note: bf16 tensor + f16 scalar or vice versa
    
    // Float16 + int -> Float16
    Tensor f16({{3}}, Dtype::Float16);
    float16_t* data_f16 = static_cast<float16_t*>(f16.data());
    data_f16[0] = float16_t(1.0f);
    data_f16[1] = float16_t(2.0f);
    data_f16[2] = float16_t(3.0f);
    Tensor result1 = f16 + 5;  // int -> int64, f16 + int64 -> ?
    std::cout << "\nFloat16 + Int32\n";
    result1.display(std::cout, 5);
    
    // Bfloat16 + int -> Bfloat16
    Tensor bf16({{3}}, Dtype::Bfloat16);
    bfloat16_t* data_bf16 = static_cast<bfloat16_t*>(bf16.data());
    data_bf16[0] = bfloat16_t(1.0f);
    data_bf16[1] = bfloat16_t(2.0f);
    data_bf16[2] = bfloat16_t(3.0f);
    Tensor result2 = bf16 + 5;
    std::cout << "\nBfloat16 + Int32\n";
    result2.display(std::cout, 5);
}

void test_commutative_operations() {
    std::cout << "\n=== Testing Scalar + Tensor (Commutativity) ===" << std::endl;
    
    Tensor a32({{3}}, Dtype::Int32);
    int32_t* data_a32 = static_cast<int32_t*>(a32.data());
    data_a32[0] = 10; data_a32[1] = 20; data_a32[2] = 30;
    
    // Tensor + Scalar
    Tensor result1 = a32 + 5.5f;
    std::cout << "\nTensor + Scalar: Int32 + Float32\n";
    result1.display(std::cout, 5);
    
    // Scalar + Tensor
    Tensor result2 = 5.5f + a32;
    std::cout << "\nScalar + Tensor: Float32 + Int32\n";
    result2.display(std::cout, 5);
    
    // Verify they're the same
    std::cout << "Results match: " << (result1.dtype() == result2.dtype() ? "YES" : "NO") << std::endl;
}

void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    // Smallest types
    Tensor a16({{2}}, Dtype::Int16);
    int16_t* data = static_cast<int16_t*>(a16.data());
    data[0] = 1; data[1] = 2;
    Tensor result1 = a16 + static_cast<int16_t>(1);
    std::cout << "\nSmallest int: Int16 + Int16\n";
    result1.display(std::cout, 5);
    
    // Largest types
    Tensor a64({{2}}, Dtype::Float64);
    double* data64 = static_cast<double*>(a64.data());
    data64[0] = 1.5; data64[1] = 2.5;
    Tensor result2 = a64 + 0.5f;
    std::cout << "\nLargest float: Float64 + Float32\n";
    result2.display(std::cout, 5);
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "========================================" << std::endl;
    std::cout << "           Promotion Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_int_int_promotion();
        test_int_float_promotion();
        test_float_float_promotion();
        test_special_cases();
        test_commutative_operations();
        test_edge_cases();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
