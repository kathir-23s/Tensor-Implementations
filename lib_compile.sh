#!/bin/bash
# Optimized build script for RTX 3060 (Ampere SM 8.6)

# Create directories if they don't exist
mkdir -p lib/objects

echo "Compiling CPU source files..."

# CPU COMPILATION
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Tensor.cpp -o lib/objects/tensor.o

# Memory Allocation
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/AllocatorRegistry.cpp -o lib/objects/AllocatorRegistry.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/CPUAllocator.cpp -o lib/objects/CPUAllocator.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/device/DeviceCore.cpp -o lib/objects/DeviceCore.o

# CUDA-compiled CPU files
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -arch=sm_86 -c src/device/CUDAAllocator.cpp -o lib/objects/CUDAAllocator.o
nvcc -std=c++20 -Iinclude -DWITH_CUDA -Xcompiler -fPIC -arch=sm_86 -c src/device/DeviceTransfer.cpp -o lib/objects/DeviceTransfer.o

# Tensor utilities
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorFactory.cpp -o lib/objects/TensorFactory.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorUtils.cpp -o lib/objects/TensorUtils.o

# Views
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Views/ViewUtils.cpp -o lib/objects/ViewUtils.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Views/ViewOps.cpp -o lib/objects/ViewOps.o

# Unary Operations (CPU)
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Arithmetics.cpp -o lib/objects/BasicArithmetic.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/ArithmeticsCore.cpp -o lib/objects/ArithmeticCore.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Exponents.cpp -o lib/objects/ExponentLog.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/ExponentCore.cpp -o lib/objects/ExponentLogCore.o

g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Trigonometry.cpp -o lib/objects/Trigonometry.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/TrigonometryDispatch.cpp -o lib/objects/TrigonometryDispatch.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/Reduction.cpp -o lib/objects/Reduction.o
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/UnaryOps/cpu/ReductionUtils.cpp -o lib/objects/ReductionUtils.o


# Scalar/Tensor Operations (CPU)
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/ScalarOps/cpu/ScalarOps.cpp -o lib/objects/ScalarOps.o

# Tensor Operations
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/TensorOps/cpu/TensorOps.cpp -o lib/objects/TensorOps.o

# Matmul and Kernels
g++ -std=c++20 -Iinclude -I/usr/include -DWITH_CUDA -fPIC -c src/Kernels/cpu/GenMatmul.cpp -o lib/objects/GEMM.o

echo "Completed CPU compilation"
echo "Compiling CUDA kernels for RTX 3060 (Ampere)..."

# ✅ CRITICAL FIX: Add -dc (device code) and ensure -fPIC is applied correctly
# Note: -Xcompiler="-fPIC" syntax works better than -Xcompiler -fPIC for some NVCC versions
NVCC_FLAGS="-std=c++20 -Iinclude -DWITH_CUDA -Xcompiler=\"-fPIC\" -arch=sm_86 -dc"

# Views
nvcc $NVCC_FLAGS -c src/Views/ContiguousKernel.cu -o lib/objects/ContiguousKernel.o

# Unary Operations (CUDA)
nvcc $NVCC_FLAGS -c src/UnaryOps/cuda/Arithmetics.cu -o lib/objects/ArithmeticsCuda.o
nvcc $NVCC_FLAGS -c src/UnaryOps/cuda/Exponents.cu -o lib/objects/ExponentLogCuda.o
nvcc $NVCC_FLAGS -c src/UnaryOps/cuda/Trigonometry.cu -o lib/objects/TrigonometryCuda.o

# ✅ REDUCTION KERNELS (with -dc flag)
nvcc $NVCC_FLAGS -c src/UnaryOps/cuda/ReductionKernels.cu -o lib/objects/ReductionKernels.o
nvcc $NVCC_FLAGS -c src/UnaryOps/cuda/ReductionImplGPU.cu -o lib/objects/ReductionImplGPU.o

# Scalar/Tensor Operations (CUDA)
nvcc $NVCC_FLAGS -c src/ScalarOps/cuda/ScalarOps.cu -o lib/objects/ScalarOpsCuda.o
nvcc $NVCC_FLAGS -c src/TensorOps/cuda/TensorOpsBrAdd.cu -o lib/objects/TensorOpsCudaAdd.o
nvcc $NVCC_FLAGS -c src/TensorOps/cuda/TensorOpsSub.cu -o lib/objects/TensorOpsCudaSub.o
nvcc $NVCC_FLAGS -c src/TensorOps/cuda/TensorOpsMul.cu -o lib/objects/TensorOpsCudaMul.o
nvcc $NVCC_FLAGS -c src/TensorOps/cuda/TensorOpsDiv.cu -o lib/objects/TensorOpsCudaDiv.o
nvcc $NVCC_FLAGS -c src/Kernels/cuda/GenMatmul.cu -o lib/objects/GEMMCuda.o

echo "Completed CUDA compilation"

# ✅ CRITICAL: Device code must be linked with -fPIC for shared library
echo "Linking CUDA device code..."
nvcc -Xcompiler="-fPIC" -arch=sm_86 -dlink \
    lib/objects/ContiguousKernel.o \
    lib/objects/ReductionKernels.o \
    lib/objects/ReductionImplGPU.o \
    -o lib/objects/device_link.o

# Link everything into static library
ar rcs lib/libtensor.a lib/objects/*.o
echo "✅ Created static library (libtensor.a)"

# ✅ FIXED: Create shared library with proper flags
echo "Creating shared library..."
nvcc -shared -Xcompiler="-fPIC" -arch=sm_86 \
    lib/objects/*.o \
    -L/usr/local/cuda/lib64 -lcudart -ltbb \
    -o lib/libtensor.so

echo "✅ Created shared library (libtensor.so)"
echo "🚀 Build complete for RTX 3060!"