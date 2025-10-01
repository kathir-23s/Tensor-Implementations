# Tensor Implementations

## Local Repository of team Tensor and Ops



## Syntax for compiling trigonometric functions
**With AVX**: ` g++ -O3 -std=c++20     -Iinclude
     local_test/test_trig.cpp     src/UnaryOps/AVX_Trigonometry.cpp     src/tensor.cpp     -o
 test_trig_no_sleef1 -ltbb   -mavx2 -mfma     -march=native`

 **Witout AVX**: ` g++ -O3 -std=c++20     -Iinclude
     local_test/test_trig.cpp     src/UnaryOps/AVX_Trigonometry.cpp     src/tensor.cpp     -o
 test_trig_no_sleef1     -ltbb     -march=native`