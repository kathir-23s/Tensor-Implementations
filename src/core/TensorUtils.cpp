// src/TensorUtils.cpp
#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"
#include "device/DeviceTransfer.h"  // for device::copy_memory (CUDA/CPU copy)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>

namespace OwnTensor {

namespace {

// ---------- pretty-print config ----------
struct PrintOptions {
    int precision  = 6;     // default print precision
    int threshold  = 1000;  // summarize if numel() > threshold
    int edgeitems  = 3;     // show these many from start/end when summarized
    int linewidth  = 120;   // not enforced strictly here, parity with numpy/torch
};

// ---------- helpers for number formatting ----------
template <typename T>
inline bool is_int_like(T v) {
    // simple heuristic: close to nearest integer in double space
    return std::abs(static_cast<double>(v) - std::round(static_cast<double>(v))) < 1e-9;
}

struct FormatInfo {
    bool   int_mode = true;   // render integers? (only when not forced to float)
    bool   sci_mode = false;  // use scientific? (decided from finite magnitudes)
    int    max_width = 1;

    // Analyze only finite values for sci decision; track width for specials too
    template <typename T>
    void analyze(const T* data, size_t n, int precision, bool force_float = false) {
        int_mode = !force_float;

        double max_abs_finite = 0.0;
        bool   has_finite     = false;
        int    special_w      = 0;   // width for "nan"/"inf"/"-inf"

        auto upd_special = [&](double x) {
            if (std::isnan(x))      special_w = std::max(special_w, 3);     // "nan"
            else if (std::isinf(x)) special_w = std::max(special_w, x > 0 ? 3 : 4); // "inf"/"-inf"
        };

        for (size_t i = 0; i < n; ++i) {
            double d = static_cast<double>(data[i]);
            if (std::isfinite(d)) {
                has_finite = true;
                double a = std::abs(d);
                if (a > max_abs_finite) max_abs_finite = a;
                if (!force_float && int_mode) {
                    if (!is_int_like(d)) int_mode = false;
                }
            } else {
                upd_special(d);
            }
        }

        if (!int_mode) {
            // choose scientific only by finite magnitudes
            sci_mode = has_finite && ((max_abs_finite >= 1e8) || (max_abs_finite > 0.0 && max_abs_finite < 1e-4));
        } else {
            sci_mode = false;
        }

        // Compute width from the largest finite representation
        int numeric_w = 1;
        if (has_finite) {
            std::ostringstream oss;
            if (int_mode) {
                oss << static_cast<long long>(std::llround(max_abs_finite));
            } else if (sci_mode) {
                oss << std::scientific << std::setprecision(precision) << max_abs_finite;
            } else {
                oss << std::fixed << std::setprecision(precision) << max_abs_finite;
            }
            numeric_w = static_cast<int>(oss.str().size());
        }

        max_width = std::max(numeric_w, special_w);
    }
};

template <typename T>
inline void format_value(std::ostream& os, T val, const FormatInfo& fmt, int precision) {
    std::ostringstream s;
    double dv = static_cast<double>(val);

    // Render IEEE-754 specials first
    if (std::isnan(dv)) {
        s << "nan";
    } else if (std::isinf(dv)) {
        s << (dv > 0 ? "inf" : "-inf");
    } else if (fmt.int_mode) {
        s << static_cast<long long>(std::llround(dv));
    } else if (fmt.sci_mode) {
        s << std::scientific << std::setprecision(precision) << dv;
    } else {
        s << std::fixed << std::setprecision(precision) << dv;
    }

    os << std::setw(fmt.max_width) << std::right << s.str();
}

// ---------- printers for concrete C++ element types ----------
template <typename T>
void print_1d(std::ostream& os, const T* ptr, size_t count, int precision,
              const PrintOptions& opts, bool force_float) {
    FormatInfo fmt;
    fmt.analyze(ptr, count, precision, /*force_float=*/force_float);

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
void print_1d_half(std::ostream& os, const HalfT* ptr, size_t count, int precision,
                   const PrintOptions& opts, ToFloatFn to_float) {
    // Convert a view to float for formatting determination
    std::vector<float> tmp;
    tmp.reserve(count);
    for (size_t i = 0; i < count; ++i) tmp.push_back(to_float(ptr[i]));

    FormatInfo fmt;
    fmt.analyze(tmp.data(), tmp.size(), precision, /*force_float=*/true);

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
void dispatch_print_1d(std::ostream& os, Dtype dt, const void* data, size_t count,
                       int precision, const PrintOptions& opts) {
    switch (dt) {
        case Dtype::Bool:{
            const uint8_t* ptr = static_cast<const uint8_t*>(data);
            const bool summarize = (count > static_cast<size_t>(opts.edgeitems * 2 + 1));
            const size_t head = summarize ? static_cast<size_t>(opts.edgeitems) : count;
            const size_t tail = summarize ? static_cast<size_t>(opts.edgeitems) : 0;
            
            for (size_t i = 0; i < head; ++i) {
                if (i) os << ", ";
                os << (ptr[i] ? "true" : "false");
            }
            
            if (summarize) {
                os << ", ..., ";
                for (size_t i = count - tail; i < count; ++i) {
                    if (i != count - tail) os << ", ";
                    os << (ptr[i] ? "true" : "false");
                }
            }
            return;
        }
        case Dtype::Int16:   return print_1d(os, static_cast<const int16_t*>(data),  count, precision, opts, /*force_float=*/false);
        case Dtype::Int32:   return print_1d(os, static_cast<const int32_t*>(data),  count, precision, opts, /*force_float=*/false);
        case Dtype::Int64:   return print_1d(os, static_cast<const int64_t*>(data),  count, precision, opts, /*force_float=*/false);

        case Dtype::Float32: return print_1d(os, static_cast<const float*>(data),    count, precision, opts, /*force_float=*/true);
        case Dtype::Float64: return print_1d(os, static_cast<const double*>(data),   count, precision, opts, /*force_float=*/true);

        case Dtype::Float16: {
            const auto* p = reinterpret_cast<const float16_t*>(data);
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

// Recursive printer over ndim for an arbitrary base pointer (data or grad)
void print_recursive_from_base(std::ostream& os,
                               const Tensor& t,
                               const void* base_ptr,
                               std::vector<int64_t>& indices,
                               int depth,
                               const PrintOptions& opts)
{
    const auto& dims    = t.shape().dims;
    const auto& strides = t.stride().strides;

    if (depth == static_cast<int>(dims.size()) - 1) {
        // Last dimension: print one contiguous line
        os << "[";
        int64_t linear = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            linear += indices[i] * strides[i];
        }
        const size_t elem_sz = t.dtype_size(t.dtype());
        const auto* base     = static_cast<const std::uint8_t*>(base_ptr);
        const void* slice    = base + static_cast<size_t>(linear) * elem_sz;

        dispatch_print_1d(os, t.dtype(), slice,
                          static_cast<size_t>(dims[depth]),
                          opts.precision, opts);
        os << "]";
        return;
    }

    // Higher dims: recurse
    os << "[";
    const int64_t dim        = dims[depth];
    const bool summarize     = (dim > opts.edgeitems * 2);
    const int64_t head       = summarize ? opts.edgeitems : dim;
    const int64_t tail_start = summarize ? (dim - opts.edgeitems) : dim;

    for (int64_t i = 0; i < head; ++i) {
        indices.push_back(i);
        print_recursive_from_base(os, t, base_ptr, indices, depth + 1, opts);
        indices.pop_back();
        if (i != head - 1 || summarize) {
            os << ",\n" << std::string(depth + 1, ' ');
        }
    }

    if (summarize) {
        os << "...,\n" << std::string(depth + 1, ' ');
        for (int64_t i = tail_start; i < dim; ++i) {
            indices.push_back(i);
            print_recursive_from_base(os, t, base_ptr, indices, depth + 1, opts);
            indices.pop_back();
            if (i != dim - 1) {
                os << ",\n" << std::string(depth + 1, ' ');
            }
        }
    }

    os << "]";
}

// Convenience: data-path recursive printer (uses tensor's own data())
void print_recursive_data(std::ostream& os,
                          const Tensor& t,
                          std::vector<int64_t>& indices,
                          int depth,
                          const PrintOptions& opts)
{
    print_recursive_from_base(os, t, t.data(), indices, depth, opts);
}

} // namespace (anon)


// ========== public: Tensor::display (data + gradient) ==========
void Tensor::display(std::ostream& os, int precision) const {
    PrintOptions opts;
    opts.precision = precision;

    // ---- Header (PyTorch-like) ----
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
    os << "'";
    if (requires_grad_) os << ", requires_grad=True";
    os << ")\n";

    // ---- Data ----
    if (shape_.dims.empty() || numel() == 0) {
        os << "[]\n";
    } else {
        std::vector<int64_t> idx;
        idx.reserve(shape_.dims.size());
        print_recursive_data(os, *this, idx, /*depth=*/0, opts);
        os << "\n";
    }

    // ---- Gradient (NEW) ----
    // Print only if a grad buffer was allocated (requires_grad_ && grad_ptr_ != nullptr)
    if (requires_grad_ && grad_ptr_) {
        os << "\nGrad(shape=(";
        for (size_t i = 0; i < shape_.dims.size(); ++i) {
            os << shape_.dims[i];
            if (i + 1 < shape_.dims.size()) os << ", ";
        }
        os << "), dtype=" << get_dtype_name(dtype_) << ", device='";
        if (device_.device == Device::CPU) os << "cpu";
        else                               os << "cuda:" << device_.index;
        os << "')\n";

        if (shape_.dims.empty() || numel() == 0) {
            os << "[]\n";
            return;
        }

        // CPU path (direct); if you enable CUDA-copy in future, handle it above.
        {
            std::vector<int64_t> idx;
            idx.reserve(shape_.dims.size());
            print_recursive_from_base(os, *this, grad_ptr_.get(), idx, /*depth=*/0, opts);
            os << "\n";
        }
    }
}

void Tensor::display() const
{
    Tensor::display(std::cout, 4);
}

void Tensor::display(int precision) const
{
    Tensor::display(std::cout, precision);
}

} // namespace OwnTensor