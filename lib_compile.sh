#!/bin/bash
# Minimal build script for tensor library

# Create directories if they don't exist
mkdir -p lib/objects

echo "Compiling CPU source files..."

# CPU COMPILATION - NO CUDA HEADERS

# Core Tensors
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Tensor.cpp -o lib/objects/tensor.o

# Memory Allocation
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/AllocatorRegistry.cpp -o lib/objects/AllocatorRegistry.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/CPUAllocator.cpp -o lib/objects/CPUAllocator.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/DeviceCore.cpp -o lib/objects/DeviceCore.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/device/CUDAAllocator.cpp -o lib/objects/CUDAAllocator.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/device/DeviceTransfer.cpp -o lib/objects/DeviceTransfer.o

# Tensor Utilities
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorFactory.cpp -o lib/objects/TensorFactory.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorUtils.cpp -o lib/objects/TensorUtils.o

# Views
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Views/ViewUtils.cpp -o lib/objects/ViewUtils.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Views/ViewOps.cpp -o lib/objects/ViewOps.o

# Unary Operations
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Arithmetics.cpp -o lib/objects/BasicArithmetic.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/ArithmeticsCore.cpp -o lib/objects/ArithmeticCore.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Exponents.cpp -o lib/objects/ExponentLog.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/ExponentCore.cpp -o lib/objects/ExponentLogCore.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Trigonometry.cpp -o lib/objects/Trigonometry.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/TrigonometryDispatch.cpp -o lib/objects/TrigonometryDispatch.o

# Scalar Operations
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/ScalarOps/cpu/ScalarOps.cpp -o lib/objects/ScalarOps.o

# Tensor Operations
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorOps/cpu/TensorOps.cpp -o lib/objects/TensorOps.o

# Matmul and Kernels
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Kernels/cpu/GenMatmul.cpp -o lib/objects/GEMM.o


echo "Completed Compiling CPU source files..."

echo "Compiling CUDA source files..."

# CUDA COMPILATION - WITH CUDA SUPPORT

nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/Views/ContiguousKernel.cu -o lib/objects/ContiguousKernel.o

nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/UnaryOps/cuda/Arithmetics.cu -o lib/objects/ArithmeticsCuda.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/UnaryOps/cuda/Exponents.cu -o lib/objects/ExponentLogCuda.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/UnaryOps/cuda/Trigonometry.cu -o lib/objects/TrigonometryCuda.o


nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/ScalarOps/cuda/ScalarOps.cu -o lib/objects/ScalarOpsCuda.o


nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/TensorOps/cuda/TensorOpsBrAdd.cu -o lib/objects/TensorOpsCudaAdd.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/TensorOps/cuda/TensorOpsSub.cu -o lib/objects/TensorOpsCudaSub.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/TensorOps/cuda/TensorOpsMul.cu -o lib/objects/TensorOpsCudaMul.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/TensorOps/cuda/TensorOpsDiv.cu -o lib/objects/TensorOpsCudaDiv.o

nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -c src/Kernels/cuda/GenMatmul.cu -o lib/objects/GEMMCuda.o


echo "Completed Compiling CPU source files..."


# Create static library
ar rcs lib/libtensor.a lib/objects/*.o
echo "Done! Created Static Library"

# Create shared library
g++ -shared -o lib/libtensor.so lib/objects/*.o -L/usr/local/cuda/lib64 -ltbb -lcudart
echo "Done! Created Shared Library"

echo "Done! Created libtensor.a and libtensor.so"