#pragma once
#include "core/Tensor.h"

namespace OwnTensor
{
 
  Tensor sin  (const Tensor&);  void sin_  (Tensor&);
  Tensor cos  (const Tensor&);  void cos_  (Tensor&);
  Tensor tan  (const Tensor&);  void tan_  (Tensor&);
  Tensor asin (const Tensor&);  void asin_ (Tensor&);
  Tensor acos (const Tensor&);  void acos_ (Tensor&);
  Tensor atan (const Tensor&);  void atan_ (Tensor&);
  Tensor sinh (const Tensor&);  void sinh_ (Tensor&);
  Tensor cosh (const Tensor&);  void cosh_ (Tensor&);
  Tensor tanh (const Tensor&);  void tanh_ (Tensor&);
  Tensor asinh(const Tensor&);  void asinh_(Tensor&);
  Tensor acosh(const Tensor&);  void acosh_(Tensor&);
  Tensor atanh(const Tensor&);  void atanh_(Tensor&);
}