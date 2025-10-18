#include <iostream>
#include "Tensor.h"   // adjust to your actual include path
#include "ScalarOps.h"

using namespace OwnTensor;
int main() {
    // --- Create a small Int32 tensor ---
    Tensor A({{3}}, Dtype::Int32);
    int32_t* dataA = static_cast<int32_t*>(A.data());
    dataA[0] = 1; dataA[1] = 2; dataA[2] = 3;

    std::cout << "Original Int32 Tensor: ";
    for (size_t i = 0; i < A.numel(); ++i)
        std::cout << dataA[i] << " ";
    std::cout << std::endl;

    // --- Case 1: Int32 tensor + Int32 scalar (no promotion) ---
    Tensor B = A + 5;
    std::cout << "Int32 + Int32 result: ";
    const int32_t* dataB = static_cast<const int32_t*>(B.data());
    for (size_t i = 0; i < B.numel(); ++i)
        std::cout << dataB[i] << " ";
    std::cout << std::endl;

    // --- Case 2: Int32 tensor + Float scalar (promote to Float32) ---
    Tensor C = A + 2.5f;
    std::cout << "Int32 + Float result: ";
    const float* dataC = static_cast<const float*>(C.data());
    for (size_t i = 0; i < C.numel(); ++i)
        std::cout << dataC[i] << " ";
    std::cout << std::endl;

    // --- Case 3: Float scalar + Int32 tensor(promote to Float32) ---
    Tensor D = 2.5f + A;
    std::cout << "Int32 + Float result: ";
    const float* dataD = static_cast<const float*>(C.data());
    for (size_t i = 0; i < C.numel(); ++i)
        std::cout << dataC[i] << " ";
    std::cout << std::endl;

    // --- Case 4: Float tensor + Float scalar ---
    Tensor E({{3}}, Dtype::Float32);
    float* dataE = static_cast<float*>(E.data());
    dataE[0] = 1.1f; dataE[1] = 2.2f; dataE[2] = 3.3f;

    Tensor F = E + 0.5f;
    std::cout << "Float32 + Float32 result: ";
    const float* dataF = static_cast<const float*>(F.data());
    for (size_t i = 0; i < F.numel(); ++i)
        std::cout << dataF[i] << " ";
    std::cout << std::endl;

    return 0;
}
