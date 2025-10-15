#include "Tensor.h"
#include <iostream>
#include <iomanip>

namespace OwnTensor
{   
    template <typename T>
    void printData(std::ostream& os, const void* data, size_t count, int precision, bool isFloat)
    {
        const T* ptr = static_cast<const T*>(data);
        for (size_t i=0; i<count; ++i)
        {
            if (isFloat)
            {
                os << std::setprecision(precision) << ptr[i];
            } else
            {
                os << ptr[i];
            }
            if (i < count - 1) os << " ";
        }
    }

    void dispatchPrint(std::ostream& os, Dtype dtype, const void* data, size_t count, int precision)
    {
        switch(dtype)
        {
            case Dtype::Int16:
                printData<int16_t>(os, data, count, precision, false); break;
            case Dtype::Int32:
                printData<int32_t>(os, data, count, precision, false); break;
            case Dtype::Int64:
                printData<int64_t>(os, data, count, precision, false); break;
            case Dtype::Float32:
                printData<float>(os, data, count, precision, false); break;
            case Dtype::Float64:
                printData<double>(os, data, count, precision, false); break;
            case Dtype::Float16:
                printData<float16_t>(os, data, count, precision, false); break;
            case Dtype::Bfloat16:
                printData<bfloat16_t>(os, data, count, precision, false); break;
            default:
                os << "<unsupported dtype>";
        }
    }

    void Tensor::display(std::ostream& os, int precision) const {
        // Print the shape of the tensor
        os << ", device=";
        if (device_.device == Device::CPU)
        {
            os << "cpu";
        } else {
            os << "cuda:" << device_.index; 
        }

        os << "Tensor(shape=[";
        const auto& dims = shape_.dims;
        for (size_t i = 0; i < dims.size(); ++i) {
            os << dims[i];
            if (i < dims.size() - 1) os << ", ";
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

        os << ")\n";

        if (dims.size() == 1) {
        os << "[";
        dispatchPrint(os, dtype_, data(), dims[0], precision);
        os << "]" << std::endl;
    } else if (dims.size() == 2) {
        os << "[\n";
        int64_t rows = dims[0];
        int64_t cols = dims[1];
        const void* base_ptr = data();
        
        for (int64_t i = 0; i < rows; ++i) {
            os << " [";
            const void* row_ptr = static_cast<const char*>(base_ptr) + i * stride_.strides[0] * dtype_size(dtype_);
            dispatchPrint(os, dtype_, row_ptr, cols, precision);
            os << "]";
            if (i < rows - 1) os << "\n";
        }
        os << "\n]" << std::endl;
    } else {
        os << "[";
        dispatchPrint(os, dtype_, data(), numel(), precision);
        os << "]" << std::endl;
    }
    }
}