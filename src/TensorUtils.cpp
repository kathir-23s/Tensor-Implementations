#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

namespace OwnTensor
{   
    template <typename T>
    void printData(std::ostream& os, const void* data, size_t count, int precision, bool isFloat)
    {
        const T* ptr = static_cast<const T*>(data);
        for (size_t i = 0; i < count; ++i)
        {
            if (isFloat)
            {
                os << std::setprecision(precision) << ptr[i];
            }
            else
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
                printData<float>(os, data, count, precision, true); break;
            case Dtype::Float64:
                printData<double>(os, data, count, precision, true); break;
            case Dtype::Float16:
                printData<float16_t>(os, data, count, precision, true); break;
            case Dtype::Bfloat16:
                printData<bfloat16_t>(os, data, count, precision, true); break;
            default:
                os << "<unsupported dtype>";
        }
    }

    void printTensorRecursive(std::ostream& os, const Tensor& tensor, 
                             const std::vector<int64_t>& indices, int depth, 
                             int precision, int max_elements = 6) 
    {
        const auto& dims = tensor.shape().dims;
        
        if (depth == dims.size() - 1) {
            // Last dimension - print the elements
            os << "[";
            
            // Calculate the offset for this slice
            int64_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                offset += indices[i] * tensor.stride().strides[i];
            }
            
            const void* slice_data = static_cast<const char*>(tensor.data()) + offset * tensor.dtype_size(tensor.dtype());
            int64_t elements_to_print = std::min(dims[depth], static_cast<int64_t>(max_elements));
            
            dispatchPrint(os, tensor.dtype(), slice_data, elements_to_print, precision);
            
            if (dims[depth] > max_elements) {
                os << " ...";
            }
            
            os << "]";
        } else {
            // Recursive case - print nested arrays
            os << "[";
            
            int64_t current_dim = dims[depth];
            int64_t elements_to_print = std::min(current_dim, static_cast<int64_t>(max_elements));
            
            for (int64_t i = 0; i < elements_to_print; ++i) {
                std::vector<int64_t> new_indices = indices;
                new_indices.push_back(i);
                
                printTensorRecursive(os, tensor, new_indices, depth + 1, precision, max_elements);
                
                if (i < elements_to_print - 1) {
                    os << "\n" << std::string(depth + 1, ' ');
                }
            }
            
            if (current_dim > max_elements) {
                os << "\n" << std::string(depth + 1, ' ') << "...";
            }
            
            if (depth == 0) {
                os << "]";
            } else {
                os << "\n" << std::string(depth, ' ') << "]";
            }
        }
    }

    void Tensor::display(std::ostream& os, int precision) const {
        // Print header like PyTorch
        os << "Tensor(";
        
        // Print shape
        const auto& dims = shape_.dims;
        os << "shape=(";
        for (size_t i = 0; i < dims.size(); ++i) {
            os << dims[i];
            if (i < dims.size() - 1) os << ", ";
        }
        os << "), ";
        
        // Print dtype using the provided function
        os << "dtype=" << get_dtype_name(dtype_) << ", ";
        
        // Print device
        os << "device='";
        if (device_.device == Device::CPU) {
            os << "cpu";
        } else {
            os << "cuda:" << device_.index;
        }
        os << "'";
        
        os << ")\n";
        
        // Print tensor data
        if (dims.empty() || numel() == 0) {
            os << "[]";
        } else {
            printTensorRecursive(os, *this, {}, 0, precision);
        }
        
        os << std::endl;
    }
}