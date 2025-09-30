#include "../include/Tensor.h"
#include <iostream>
#include <iomanip>

void Tensor::display(std::ostream& os) const {
    // Print the shape of the tensor
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.dims.size(); ++i) {
        os << shape_.dims[i];
        if (i < shape_.dims.size() - 1) os << ", ";
    }

    // Print the datatype of the tensor
    os << "], dtype=";
    switch(dtype_)
    {
        case Dtype::Int16: os << "Int16"; break;
        case Dtype::Int32: os << "Int32"; break;
        case Dtype::Int64: os << "Int64"; break;
        case Dtype::Float16: os << "Float16"; break;
        case Dtype::Float32: os << "Float32"; break;
        case Dtype::Float64: os << "Float64"; break;
        case Dtype::Bfloat16: os << "Bfloat16"; break;
    }

    // Printing device of the tensor
    os << ", device=";
    if (device_.device == Device::CPU)
    {
        os << "cpu";
    } else {
        os << "cuda:" << device_.index; 
    }

    os << ")\n";

    if (shape_.dims.size() == 1) {
        os << "[";
        if (dtype_ == Dtype::Int32) {
            const int32_t* ptr = static_cast<const int32_t*>(data());
            for (int64_t i = 0; i < shape_.dims[0]; ++i) {
                os << ptr[i] << " ";
            }
        } else if (dtype_ == Dtype::Float32) {
            const float* ptr = static_cast<const float*>(data());
            for (int64_t i = 0; i < shape_.dims[0]; ++i) {
                os << ptr[i] << " ";
            }
        }
        os << "]" << std::endl;
    } else if (shape_.dims.size() == 2) {
        os << "[\n";
        int64_t rows = shape_.dims[0];
        int64_t cols = shape_.dims[1];
        if (dtype_ == Dtype::Int32) {
            const int32_t* ptr = static_cast<const int32_t*>(data());
            for (int64_t i = 0; i < rows; ++i) {
                os << " [";
                for (int64_t j = 0; j < cols; ++j) {
                    int64_t offset = i * stride_.strides[0] + j * stride_.strides[1];
                    os << ptr[offset];
                    if (j < cols - 1) os << " ";
                }
                os << "]";
                if (i < rows - 1) os << "\n";
            }
        } else if (dtype_ == Dtype::Float32) {
            const float* ptr = static_cast<const float*>(data());
            for (int64_t i = 0; i < rows; ++i) {
                os << " [";
                for (int64_t j = 0; j < cols; ++j) {
                    int64_t offset = i * stride_.strides[0] + j * stride_.strides[1];
                    os << ptr[offset];
                    if (j < cols - 1) os << " ";
                }
                os << "]";
                if (i < rows - 1) os << "\n";
            }
        }
        os << "\n]" << std::endl;
    } else {
        // For tensors with ndim > 2, print elements in a flat sequence
        os << "[";
        size_t total_elems = numel();
        if (dtype_ == Dtype::Int32) {
            const int32_t* ptr = static_cast<const int32_t*>(data());
            for (size_t i = 0; i < total_elems; ++i) {
                os << ptr[i];
                if (i < total_elems - 1) os << ", ";
            }
        } else if (dtype_ == Dtype::Float32) {
            const float* ptr = static_cast<const float*>(data());
            for (size_t i = 0; i < total_elems; ++i) {
                os << ptr[i];
                if (i < total_elems - 1) os << ", ";
            }
        }
        os << "]" << std::endl;
    }
}
