// src/TensorUtils.cpp
#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace OwnTensor {

namespace {

// ---------- pretty-print config ----------
struct PrintOptions {
    int precision  = 6;     // default print precision
    int threshold  = 1000;  // summarize if numel() > threshold
    int edgeitems  = 3;     // show these many from start/end when summarized
    int linewidth  = 120;   // not enforced strictly here, but kept for parity
};

// ---------- helpers for number formatting ----------
template <typename T>
inline bool is_int_like(T v) {
    // simple heuristic: close to nearest integer in double space
    return std::abs(static_cast<double>(v) - std::round(static_cast<double>(v))) < 1e-9;
}

struct FormatInfo {
    bool int_mode = true;   // all values integer-like?
    bool sci_mode = false;  // use scientific?
    int  max_width = 1;
    double max_abs = 0.0;

    template <typename T>
    void analyze(const T* data, size_t n, int precision) {
        int_mode = true;
        max_abs = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double a = std::abs(static_cast<double>(data[i]));
            if (a > max_abs) max_abs = a;
            if (int_mode && !is_int_like(data[i])) {
                int_mode = false;
            }
        }

        if (!int_mode) {
            sci_mode = (max_abs >= 1e8) || (max_abs > 0.0 && max_abs < 1e-4);
        } else {
            sci_mode = false;
        }

        std::ostringstream oss;
        if (int_mode) {
            oss << static_cast<long long>(std::llround(max_abs));
        } else if (sci_mode) {
            oss << std::scientific << std::setprecision(precision) << max_abs;
        } else {
            oss << std::fixed << std::setprecision(precision) << max_abs;
        }
        max_width = std::max<int>(static_cast<int>(oss.str().size()), 1);
    }
};

// template <typename T>
// inline void format_value(std::ostream& os, T v, const FormatInfo& fmt, int precision) {
//     std::ostringstream s;
//     if (fmt.int_mode) {
//         s << static_cast<long long>(std::llround(static_cast<double>(v)));
//     } else if (fmt.sci_mode) {
//         s << std::scientific << std::setprecision(precision) << static_cast<double>(v);
//     } else {
//         s << std::fixed << std::setprecision(precision) << static_cast<double>(v);
//     }
//     os << std::setw(fmt.max_width) << std::right << s.str();
// }
template <typename T>
inline void format_value(std::ostream& os, T val, const FormatInfo& fmt, int precision) {
    std::ostringstream s;

    // // Convert bfloat16 to float before checks if applicable:
    // float val = static_cast<float>(v);

    if (std::isnan(val)) {
        s << "nan";
    } else if (std::isinf(val)) {
        if (val > 0)
            s << "inf";
        else
            s << "-inf";
    } else if (fmt.int_mode) {
        s << static_cast<long long>(std::llround(static_cast<double>(val)));
    } else if (fmt.sci_mode) {
        s << std::scientific << std::setprecision(precision) << val;
    } else {
        s << std::fixed << std::setprecision(precision) << val;
    }

    os << std::setw(fmt.max_width) << std::right << s.str();
}


// ---------- printers for concrete C++ element types ----------
template <typename T>
void print_1d(std::ostream& os, const T* ptr, size_t count, int precision, const PrintOptions& opts) {
    FormatInfo fmt;
    fmt.analyze(ptr, count, precision);

    const bool summarize = (count > static_cast<size_t>(opts.edgeitems * 2 + 1));
    const size_t head = summarize ? static_cast<size_t>(opts.edgeitems) : count;
    const size_t tail = summarize ? static_cast<size_t>(opts.edgeitems) : 0;

    for (size_t i = 0; i < head; ++i) {
        if (i) os << ", ";
        format_value(os, ptr[i], fmt, precision);
    }

    if (summarize) {
        os << ", ..., ";
        for (size_t i = count - tail; i < count; ++i) {
            if (i != count - tail) os << ", ";
            format_value(os, ptr[i], fmt, precision);
        }
    }
}

// Convert-and-print path for half types stored as custom wrappers
template <typename HalfT, typename ToFloatFn>
void print_1d_half(std::ostream& os, const HalfT* ptr, size_t count, int precision, const PrintOptions& opts, ToFloatFn to_float) {
    // Convert a view to float for formatting determination
    std::vector<float> tmp;
    tmp.reserve(count);
    for (size_t i = 0; i < count; ++i) tmp.push_back(to_float(ptr[i]));

    FormatInfo fmt;
    fmt.analyze(tmp.data(), tmp.size(), precision);

    const bool summarize = (count > static_cast<size_t>(opts.edgeitems * 2 + 1));
    const size_t head = summarize ? static_cast<size_t>(opts.edgeitems) : count;
    const size_t tail = summarize ? static_cast<size_t>(opts.edgeitems) : 0;

    for (size_t i = 0; i < head; ++i) {
        if (i) os << ", ";
        format_value(os, tmp[i], fmt, precision);
    }
    if (summarize) {
        os << ", ..., ";
        for (size_t i = count - tail; i < count; ++i) {
            if (i != count - tail) os << ", ";
            format_value(os, tmp[i], fmt, precision);
        }
    }
}

// Dispatch to a concrete print implementation by dtype.
// Expects a pointer to the start of the contiguous slice (CPU-accessible).
void dispatch_print_1d(std::ostream& os, Dtype dt, const void* data, size_t count, int precision, const PrintOptions& opts) {
    switch (dt) {
        case Dtype::Int16:   return print_1d(os, static_cast<const int16_t*>(data),  count, precision, opts);
        case Dtype::Int32:   return print_1d(os, static_cast<const int32_t*>(data),  count, precision, opts);
        case Dtype::Int64:   return print_1d(os, static_cast<const int64_t*>(data),  count, precision, opts);
        case Dtype::Float32: return print_1d(os, static_cast<const float*>(data),    count, precision, opts);
        case Dtype::Float64: return print_1d(os, static_cast<const double*>(data),   count, precision, opts);

        case Dtype::Float16: {
            const auto* p = reinterpret_cast<const float16_t*>(data);
            // convert via detail::float16_to_float on the raw bits
            auto to_float = [](float16_t h) -> float {
                return detail::float16_to_float(h.raw_bits);
            };
            return print_1d_half(os, p, count, precision, opts, to_float);
        }

        case Dtype::Bfloat16: {
            const auto* p = reinterpret_cast<const bfloat16_t*>(data);
            auto to_float = [](bfloat16_t b) -> float {
                return detail::bfloat16_to_float(b.raw_bits);
            };
            return print_1d_half(os, p, count, precision, opts, to_float);
        }

        default:
            os << "<unsupported dtype>";
            return;
    }
}

// Recursive printer over ndim
void print_recursive(std::ostream& os,
                     const Tensor& t,
                     std::vector<int64_t>& indices,
                     int depth,
                     const PrintOptions& opts)
{
    const auto& dims = t.shape().dims;
    const auto& strides = t.stride().strides;

    if (depth == static_cast<int>(dims.size()) - 1) {
        // Last dimension: print one contiguous line
        os << "[";
        // Compute byte offset of this slice
        int64_t linear = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            linear += indices[i] * strides[i];
        }
        const size_t elem_sz = t.dtype_size(t.dtype());
        const auto* base = static_cast<const std::uint8_t*>(t.data());
        const void* slice = base + static_cast<size_t>(linear) * elem_sz;

        dispatch_print_1d(os, t.dtype(), slice, static_cast<size_t>(dims[depth]), opts.precision, opts);
        os << "]";
        return;
    }

    // Higher dims: recurse
    os << "[";
    const int64_t dim = dims[depth];
    const bool summarize = (dim > opts.edgeitems * 2);
    const int64_t head = summarize ? opts.edgeitems : dim;
    const int64_t tail_start = summarize ? (dim - opts.edgeitems) : dim;

    for (int64_t i = 0; i < head; ++i) {
        indices.push_back(i);
        print_recursive(os, t, indices, depth + 1, opts);
        indices.pop_back();
        if (i != head - 1 || summarize) {
            os << ",\n" << std::string(depth + 1, ' ');
        }
    }

    if (summarize) {
        os << "...,\n" << std::string(depth + 1, ' ');
        for (int64_t i = tail_start; i < dim; ++i) {
            indices.push_back(i);
            print_recursive(os, t, indices, depth + 1, opts);
            indices.pop_back();
            if (i != dim - 1) {
                os << ",\n" << std::string(depth + 1, ' ');
            }
        }
    }

    os << "]";
}

} // namespace (anon)

// ========== public: Tensor::display ==========
void Tensor::display(std::ostream& os, int precision) const {
    PrintOptions opts;
    opts.precision = precision;

    // Header like PyTorch
    os << "Tensor(shape=(";
    for (size_t i = 0; i < shape_.dims.size(); ++i) {
        os << shape_.dims[i];
        if (i + 1 < shape_.dims.size()) os << ", ";
    }
    os << "), dtype=" << get_dtype_name(dtype_) << ", device='";
    if (device_.device == Device::CPU) {
        os << "cpu";
    } else {
        os << "cuda:" << device_.index;
    }
    os << "')\n";

    // Optional debug: dump first element raw bits for half types
    if (numel() > 0) {
        if (dtype_ == Dtype::Float16) {
            auto* p = reinterpret_cast<const float16_t*>(this->data());
            // os << "[debug] first f16 raw_bits=0x"
            //    << std::hex << std::setw(4) << std::setfill('0')
            //    << static_cast<unsigned>(p[0].raw_bits)
            //    << std::dec << " (expect 0x3c00 for 1.0)\n";
        } else if (dtype_ == Dtype::Bfloat16) {
            auto* p = reinterpret_cast<const bfloat16_t*>(this->data());
            // os << "[debug] first bf16 raw_bits=0x"
            //    << std::hex << std::setw(4) << std::setfill('0')
            //    << static_cast<unsigned>(p[0].raw_bits)
            //    << std::dec << " (expect 0x3f80 for 1.0)\n";
        }
    }

    // Data
    if (shape_.dims.empty() || numel() == 0) {
        os << "[]\n";
        return;
    }

    std::vector<int64_t> idx;
    idx.reserve(shape_.dims.size());
    print_recursive(os, *this, idx, /*depth=*/0, opts);
    os << "\n";
}

} // namespace OwnTensor