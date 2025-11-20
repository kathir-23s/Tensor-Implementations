#pragma once
#ifndef TOPS_LIB_H
#define TOPS_LIB_H

// Core Tensor Interface
#include "core/Tensor.h"


// Operations
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Trigonometry.h"

#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"

#include "mlp/activations.h"

#endif // TOPS_LIB_H