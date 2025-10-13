#!/bin/bash
# Minimal build script for tensor library

# Compile source file into object
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -DWITH_DEBUG -fPIC -c src/Tensor.cpp -o lib/objects/tensor.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/AllocatorRegistry.cpp -o lib/objects/AllocatorRegistry.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/CPUAllocator.cpp -o lib/objects/CPUAllocator.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/CUDAAllocator.cpp -o lib/objects/CUDAAllocator.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/DeviceCore.cpp -o lib/objects/DeviceCore.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/DeviceTransfer.cpp -o lib/objects/DeviceTransfer.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorFactory.cpp -o lib/objects/TensorFactory.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorUtils.cpp -o lib/objects/TensorUtils.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/Trigonometry.cpp -o lib/objects/Trigonometry.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/Exponents.cpp -o lib/objects/ExponentLog.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/Arithmetics.cpp -o lib/objects/BasicArithmetic.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/UnaryUnifiedDispatcher.cpp -o lib/objects/UnaryDispatcher.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/Reduction.cpp -o lib/objects/Reduction.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/ReductionUtils.cpp -o lib/objects/ReductionUtils.o

# Create static library
ar rcs lib/libtensor.a lib/objects/*.o

# Create shared library
g++ -shared -o lib/libtensor.so lib/objects/*.o -ltbb

echo "Done! Created libtensor.a and libtensor.so"
