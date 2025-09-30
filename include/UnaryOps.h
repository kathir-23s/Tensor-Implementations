#pragma once
#include "/home/blubridge-036/Madhu_folder/Tensor-Implementations/include/Tensor.h"
#include <iostream>
#include <cmath>
#include <cstdint>
#include <stdexcept>

using namespace std;

// basic arithmetics
//in place
void square_(Tensor& t);
void square_root_(Tensor& t);
void power_(Tensor& t, int exponent);
void neg_(Tensor& t);       
void abs_(Tensor& t);   
void sign_(Tensor& t);

//out of place
Tensor square(const Tensor& t);
Tensor square_root(const Tensor& t);
Tensor power(const Tensor& t, int exponent);
Tensor neg(const Tensor& t); 
Tensor abs(const Tensor& t);
Tensor sign(const Tensor& t);
// trignometrics









// exponentials and logarithms









// data operations