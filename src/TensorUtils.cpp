// #include "core/Tensor.h"
// #include "dtype/DtypeTraits.h"
// #include <iostream>
// #include <iomanip>
// #include <vector>
// #include <algorithm>

// namespace OwnTensor
// {   
//     template <typename T>
//     void printData(std::ostream& os, const void* data, size_t count, int precision, bool isFloat)
//     {
//         const T* ptr = static_cast<const T*>(data);
//         for (size_t i = 0; i < count; ++i)
//         {
//             if (isFloat)
//             {
//                 os << std::setprecision(precision) << ptr[i];
//             }
//             else
//             {
//                 os << ptr[i];
//             }
//             if (i < count - 1) os << " ";
//         }
//     }

//     void dispatchPrint(std::ostream& os, Dtype dtype, const void* data, size_t count, int precision)
//     {
//         switch(dtype)
//         {
//             case Dtype::Int16:
//                 printData<int16_t>(os, data, count, precision, false); break;
//             case Dtype::Int32:
//                 printData<int32_t>(os, data, count, precision, false); break;
//             case Dtype::Int64:
//                 printData<int64_t>(os, data, count, precision, false); break;
//             case Dtype::Float32:
//                 printData<float>(os, data, count, precision, true); break;
//             case Dtype::Float64:
//                 printData<double>(os, data, count, precision, true); break;
//             case Dtype::Float16:
//                 printData<float16_t>(os, data, count, precision, true); break;
//             case Dtype::Bfloat16:
//                 printData<bfloat16_t>(os, data, count, precision, true); break;
//             default:
//                 os << "<unsupported dtype>";
//         }
//     }

//     void printTensorRecursive(std::ostream& os, const Tensor& tensor, 
//                              const std::vector<int64_t>& indices, int depth, 
//                              int precision, int max_elements = 6) 
//     {
//         const auto& dims = tensor.shape().dims;
        
//         if (depth == dims.size() - 1) {
//             // Last dimension - print the elements
//             os << "[";
            
//             // Calculate the offset for this slice
//             int64_t offset = 0;
//             for (size_t i = 0; i < indices.size(); ++i) {
//                 offset += indices[i] * tensor.stride().strides[i];
//             }
            
//             const void* slice_data = static_cast<const char*>(tensor.data()) + offset * tensor.dtype_size(tensor.dtype());
//             int64_t elements_to_print = std::min(dims[depth], static_cast<int64_t>(max_elements));
            
//             dispatchPrint(os, tensor.dtype(), slice_data, elements_to_print, precision);
            
//             if (dims[depth] > max_elements) {
//                 os << " ...";
//             }
            
//             os << "]";
//         } else {
//             // Recursive case - print nested arrays
//             os << "[";
            
//             int64_t current_dim = dims[depth];
//             int64_t elements_to_print = std::min(current_dim, static_cast<int64_t>(max_elements));
            
//             for (int64_t i = 0; i < elements_to_print; ++i) {
//                 std::vector<int64_t> new_indices = indices;
//                 new_indices.push_back(i);
                
//                 printTensorRecursive(os, tensor, new_indices, depth + 1, precision, max_elements);
                
//                 if (i < elements_to_print - 1) {
//                     os << "\n" << std::string(depth + 1, ' ');
//                 }
//             }
            
//             if (current_dim > max_elements) {
//                 os << "\n" << std::string(depth + 1, ' ') << "...";
//             }
            
//             if (depth == 0) {
//                 os << "]";
//             } else {
//                 os << "\n" << std::string(depth, ' ') << "]";
//             }
//         }
//     }

//     void Tensor::display(std::ostream& os, int precision) const {
//         // Print header like PyTorch
//         os << "Tensor(";
        
//         // Print shape
//         const auto& dims = shape_.dims;
//         os << "shape=(";
//         for (size_t i = 0; i < dims.size(); ++i) {
//             os << dims[i];
//             if (i < dims.size() - 1) os << ", ";
//         }
//         os << "), ";
        
//         // Print dtype using the provided function
//         os << "dtype=" << get_dtype_name(dtype_) << ", ";
        
//         // Print device
//         os << "device='";
//         if (device_.device == Device::CPU) {
//             os << "cpu";
//         } else {
//             os << "cuda:" << device_.index;
//         }
//         os << "'";
        
//         os << ")\n";
        
//         // Print tensor data
//         if (dims.empty() || numel() == 0) {
//             os << "[]";
//         } else {
//             printTensorRecursive(os, *this, {}, 0, precision);
//         }
        
//         os << std::endl;
//     }
// }

#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace OwnTensor {

// PyTorch-style print options
struct PrintOptions {
    int precision = 4;
    int threshold = 1000;    // Summarize if numel() > threshold
    int edgeitems = 3;       // Items to show at start/end when summarizing
    int linewidth = 80;      // Max characters per line
};

// Helper to determine if a float value looks like an integer
template<typename T>
bool is_int_like(T val) {
    return std::abs(val - std::round(val)) < 1e-6;
}

// Analyze data to determine formatting parameters
struct FormatInfo {
    bool int_mode = true;
    bool sci_mode = false;
    int max_width = 1;
    double max_abs = 0.0;
    
    template<typename T>
    void analyze(const T* data, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            T val = data[i];
            double abs_val = std::abs(static_cast<double>(val));
            
            if (abs_val > max_abs) max_abs = abs_val;
            
            // Check if all values are integer-like
            if (int_mode && !is_int_like(val)) {
                int_mode = false;
            }
        }
        
        // Decide scientific notation
        if (!int_mode) {
            sci_mode = (max_abs >= 1e8 || (max_abs > 0 && max_abs < 1e-4));
        }
        
        // Calculate max width needed
        std::ostringstream oss;
        if (int_mode) {
            oss << static_cast<int64_t>(max_abs);
        } else if (sci_mode) {
            oss << std::scientific << std::setprecision(4) << max_abs;
        } else {
            oss << std::fixed << std::setprecision(4) << max_abs;
        }
        max_width = std::max(static_cast<int>(oss.str().length()), max_width);
    }
};

// Format a single value with proper width and alignment
template<typename T>
void formatValue(std::ostream& os, T val, const FormatInfo& fmt, int precision) {
    std::ostringstream oss;
    
    if (fmt.int_mode) {
        oss << static_cast<int64_t>(val);
    } else if (fmt.sci_mode) {
        oss << std::scientific << std::setprecision(precision) << val;
    } else {
        oss << std::fixed << std::setprecision(precision) << val;
    }
    
    // Right-align the output
    os << std::setw(fmt.max_width) << std::right << oss.str();
}

// Print data with proper formatting
template<typename T>
void printData(std::ostream& os, const void* data, size_t count, 
               int precision, bool isFloat, const PrintOptions& opts) {
    const T* ptr = static_cast<const T*>(data);
    
    // Analyze data for formatting
    FormatInfo fmt;
    if (isFloat) {
        fmt.analyze(ptr, count);
    } else {
        fmt.int_mode = true;
        // Calculate max width for integers
        int64_t max_val = 0;
        for (size_t i = 0; i < count; ++i) {
            int64_t val = static_cast<int64_t>(ptr[i]);
            if (std::abs(val) > max_val) max_val = std::abs(val);
        }
        std::ostringstream oss;
        oss << max_val;
        fmt.max_width = oss.str().length();
    }
    
    // Check if we should summarize (for 1D arrays)
    bool summarize = count > static_cast<size_t>(opts.edgeitems * 2 + 1);
    size_t items_to_print = summarize ? opts.edgeitems : count;
    
    for (size_t i = 0; i < items_to_print; ++i) {
        if (isFloat) {
            formatValue(os, ptr[i], fmt, precision);
        } else {
            os << std::setw(fmt.max_width) << std::right << ptr[i];
        }
        
        if (i < items_to_print - 1) os << ", ";
    }
    
    if (summarize) {
        os << ", ..., ";
        // Print last edgeitems
        for (size_t i = count - opts.edgeitems; i < count; ++i) {
            if (isFloat) {
                formatValue(os, ptr[i], fmt, precision);
            } else {
                os << std::setw(fmt.max_width) << std::right << ptr[i];
            }
            if (i < count - 1) os << ", ";
        }
    }
}

void dispatchPrint(std::ostream& os, Dtype dtype, const void* data, size_t count, 
                   int precision, const PrintOptions& opts) {
    switch(dtype) {
        case Dtype::Int16:
            printData<int16_t>(os, data, count, precision, false, opts); break;
        case Dtype::Int32:
            printData<int32_t>(os, data, count, precision, false, opts); break;
        case Dtype::Int64:
            printData<int64_t>(os, data, count, precision, false, opts); break;
        case Dtype::Float32:
            printData<float>(os, data, count, precision, true, opts); break;
        case Dtype::Float64:
            printData<double>(os, data, count, precision, true, opts); break;
        case Dtype::Float16:
            printData<float>(os, data, count, precision, true, opts); break;
        case Dtype::Bfloat16:
            printData<float>(os, data, count, precision, true, opts); break;
        default:
            os << "<unsupported dtype>";
    }
}

void printTensorRecursive(std::ostream& os, const Tensor& tensor,
                         const std::vector<int64_t>& indices, int depth,
                         const PrintOptions& opts) {
    const auto& dims = tensor.shape().dims;
    
    if (depth == dims.size() - 1) {
        // Last dimension - print the elements
        os << "[";
        
        // Calculate the offset for this slice
        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * tensor.stride().strides[i];
        }
        
        const void* slice_data = static_cast<const char*>(tensor.data()) + 
                                offset * tensor.dtype_size(tensor.dtype());
        
        dispatchPrint(os, tensor.dtype(), slice_data, dims[depth], opts.precision, opts);
        os << "]";
    } else {
        // Recursive case - print nested arrays
        os << "[";
        int64_t current_dim = dims[depth];
        
        // Check if we should summarize this dimension
        bool summarize = current_dim > static_cast<int64_t>(opts.edgeitems * 2);
        int64_t items_to_print = summarize ? opts.edgeitems : current_dim;
        
        for (int64_t i = 0; i < items_to_print; ++i) {
            std::vector<int64_t> new_indices = indices;
            new_indices.push_back(i);
            printTensorRecursive(os, tensor, new_indices, depth + 1, opts);
            
            if (i < items_to_print - 1 || summarize) {
                os << ",\n" << std::string(depth + 1, ' ');
            }
        }
        
        if (summarize) {
            os << "...,\n" << std::string(depth + 1, ' ');
            // Print last edgeitems
            for (int64_t i = current_dim - opts.edgeitems; i < current_dim; ++i) {
                std::vector<int64_t> new_indices = indices;
                new_indices.push_back(i);
                printTensorRecursive(os, tensor, new_indices, depth + 1, opts);
                if (i < current_dim - 1) {
                    os << ",\n" << std::string(depth + 1, ' ');
                }
            }
        }
        
        os << "]";
    }
}

void Tensor::display(std::ostream& os, int precision) const {
    PrintOptions opts;
    opts.precision = precision;
    
    // Print tensor info header
    os << "Tensor(shape=(";
    const auto& dims = shape_.dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        os << dims[i];
        if (i < dims.size() - 1) os << ", ";
    }
    os << "), dtype=" << get_dtype_name(dtype_);
    
    os << ", device='";
    if (device_.device == Device::CPU) {
        os << "cpu";
    } else {
        os << "cuda:" << device_.index;
    }
    os << "')\n";
    
    // Print tensor data
    if (shape_.dims.empty() || numel() == 0) {
        os << "[]";
    } else {
        printTensorRecursive(os, *this, {}, 0, opts);
    }
    
    os << std::endl;
}

} // namespace OwnTensor