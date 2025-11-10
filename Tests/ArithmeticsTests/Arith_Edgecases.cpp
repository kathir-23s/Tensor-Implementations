#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "ops/helpers/testutils.h"

using namespace OwnTensor;
using namespace TestUtils;

// ============================================================================
// Tolerance Helpers
// ============================================================================

struct Tolerances { double rel_tol; double abs_tol; };

static Tolerances tol_for(Dtype dt) {
    switch (dt) {
        case Dtype::Float16:
        case Dtype::Bfloat16: return {5e-2, 1e-3};
        case Dtype::Float32: return {1e-5, 1e-6};
        case Dtype::Float64: return {1e-7, 1e-9};
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64: return {0.0, 0.5};  // Integer tolerance
        default: return {1e-5, 1e-6};
    }
}

static bool almost_equal(double a, double b, const Tolerances& t) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    double diff = std::fabs(a - b);
    if (diff <= t.abs_tol) return true;
    double maxab = std::max(std::fabs(a), std::fabs(b));
    return diff <= t.rel_tol * maxab;
}

// ============================================================================
// Tensor Creation Helpers
// ============================================================================

static Tensor make_tensor_cpu(const std::vector<double>& vals, Dtype dt, const std::vector<int64_t>& shape) {
    Tensor t(Shape{shape}, dt, DeviceIndex(Device::CPU), false);
    switch (dt) {
        case Dtype::Int16: {
            auto* p = t.data<int16_t>();
            for (size_t i=0; i<vals.size(); ++i) p[i] = static_cast<int16_t>(vals[i]);
            break;
        }
        case Dtype::Int32: {
            auto* p = t.data<int32_t>();
            for (size_t i=0; i<vals.size(); ++i) p[i] = static_cast<int32_t>(vals[i]);
            break;
        }
        case Dtype::Int64: {
            auto* p = t.data<int64_t>();
            for (size_t i=0; i<vals.size(); ++i) p[i] = static_cast<int64_t>(vals[i]);
            break;
        }
        case Dtype::Float32: {
            auto* p = t.data<float>();
            for (size_t i=0; i<vals.size(); ++i) p[i] = static_cast<float>(vals[i]);
            break;
        }
        case Dtype::Float64: {
            auto* p = t.data<double>();
            for (size_t i=0; i<vals.size(); ++i) p[i] = vals[i];
            break;
        }
        default: break;
    }
    return t;
}

static std::vector<double> to_double_vec(const Tensor& t) {
    std::vector<double> out(t.numel());
    switch (t.dtype()) {
        case Dtype::Int16: { auto* p = t.data<int16_t>(); for (int i=0; i<(int)out.size(); ++i) out[i] = static_cast<double>(p[i]); break; }
        case Dtype::Int32: { auto* p = t.data<int32_t>(); for (int i=0; i<(int)out.size(); ++i) out[i] = static_cast<double>(p[i]); break; }
        case Dtype::Int64: { auto* p = t.data<int64_t>(); for (int i=0; i<(int)out.size(); ++i) out[i] = static_cast<double>(p[i]); break; }
        case Dtype::Float32: { auto* p = t.data<float>(); for (int i=0; i<(int)out.size(); ++i) out[i] = static_cast<double>(p[i]); break; }
        case Dtype::Float64: { auto* p = t.data<double>(); for (int i=0; i<(int)out.size(); ++i) out[i] = p[i]; break; }
        default: break;
    }
    return out;
}

// ============================================================================
// Reference Implementations
// ============================================================================

static std::vector<double> ref_unary(const std::vector<double>& x, const std::string& op) {
    std::vector<double> y(x.size());
    if (op=="square") for (size_t i=0; i<x.size(); ++i) y[i] = x[i] * x[i];
    else if (op=="sqrt") for (size_t i=0; i<x.size(); ++i) y[i] = std::sqrt(x[i]);
    else if (op=="reciprocal") for (size_t i=0; i<x.size(); ++i) y[i] = 1.0 / x[i];
    else if (op=="negate") for (size_t i=0; i<x.size(); ++i) y[i] = -x[i];
    else if (op=="abs") for (size_t i=0; i<x.size(); ++i) y[i] = std::fabs(x[i]);
    else if (op=="sign") for (size_t i=0; i<x.size(); ++i) {
        y[i] = (x[i] > 0.0) ? 1.0 : ((x[i] < 0.0) ? -1.0 : 0.0);
    }
    return y;
}

static Tensor apply_unary(const Tensor& a, const std::string& op) {
    if (op=="square") return square(a,0);
    if (op=="sqrt") return sqrt(a, 0);
    if (op=="reciprocal") return reciprocal(a, 0);
    if (op=="negate") return neg(a, 0);
    if (op=="abs") return abs(a, 0);
    if (op=="sign") return sign(a, 0);
    throw std::runtime_error("Unknown op: " + op);
}

static void apply_unary_inplace(Tensor& a, const std::string& op) {
    if (op=="square") square_(a, 0);
    else if (op=="sqrt") sqrt_(a, 0);
    else if (op=="reciprocal") reciprocal_(a, 0);
    else if (op=="negate") neg_(a, 0);
    else if (op=="abs") abs_(a, 0);
    else if (op=="sign") sign_(a, 0);
    else throw std::runtime_error("Unknown op: " + op);
}

static bool check_tensor(const Tensor& out, const std::vector<double>& ref, const Tolerances& tol, std::string& msg) {
    auto got = to_double_vec(out);
    for (size_t i=0; i<ref.size(); ++i) {
        if (!almost_equal(got[i], ref[i], tol)) {
            std::ostringstream oss;
            oss << "Mismatch at index " << i << ": got=" << got[i] << ", expected=" << ref[i];
            msg = oss.str();
            return false;
        }
    }
    msg = "passed";
    return true;
}

// ============================================================================
// Test Report Structure
// ============================================================================

struct EdgeTestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double execution_time_ms;
};

class EdgeReport {
    std::vector<EdgeTestResult> results;
public:
    void add(EdgeTestResult r) { results.push_back(r); }
    
    void write() {
        std::ofstream f("local_test/arith_edge_report.md");
        f << "# Arithmetic Edge Cases Test Report\n\n";
        
        int pass=0, fail=0;
        for (auto& r: results) r.passed ? ++pass : ++fail;
        
        f << "## Summary\n";
        f << "- Total: " << results.size() << "\n";
        f << "- Passed: " << pass << "\n";
        f << "- Failed: " << fail << "\n\n";
        
        f << "## Results\n\n";
        for (auto& r: results) {
            f << (r.passed ? "✅" : "❌") << " **" << r.test_name << "** ";
            f << "(" << std::fixed << std::setprecision(3) << r.execution_time_ms << " ms)\n";
            if (!r.message.empty()) f << "   - " << r.message << "\n";
        }
        f.close();
    }
};

// ============================================================================
// MAIN - Edge Case Tests
// ============================================================================

int main() {
    EdgeReport report;
    
    std::vector<std::string> ops = {"square", "sqrt", "reciprocal", "negate", "abs", "sign"};
    std::vector<Dtype> dtypes = {Dtype::Float32, Dtype::Float64};
    
    std::cout << "\n========================================\n";
    std::cout << " ARITHMETIC EDGE CASES TEST\n";
    std::cout << "========================================\n\n";
    
    // ========================================================================
    // 1) Special Values (Infinity, NaN, ±0)
    // ========================================================================
    
    std::cout << "Testing special values...\n";
    for (auto dt: dtypes) {
        for (auto& op: ops) {
            std::vector<double> x = { 
                +0.0, -0.0,
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN() 
            };
            
            auto ref = ref_unary(x, op);
            auto t0 = std::chrono::high_resolution_clock::now();
            Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
            Tensor y = apply_unary(a, op);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
            report.add({ "specials/" + op + "(" + get_dtype_name(dt) + ")", ok, msg,
                std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }
    
    // ========================================================================
    // 2) Domain Violations (sqrt/reciprocal of negative/zero)
    // ========================================================================
    
    std::cout << "Testing domain violations...\n";
    {
        std::vector<double> x = {-4.0, -1.0, -0.0, +0.0, 1.0, 4.0};
        for (auto dt: dtypes) {
            // SQRT of negative numbers
            {
                auto ref = ref_unary(x, "sqrt");
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
                Tensor y = sqrt(a, 0);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
                report.add({ "domain/sqrt_negative(" + get_dtype_name(dt) + ")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
            
            // RECIPROCAL of zero (should give ±inf)
            {
                auto ref = ref_unary(x, "reciprocal");
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
                Tensor y = reciprocal(a, 0);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
                report.add({ "domain/reciprocal_zero(" + get_dtype_name(dt) + ")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }
    
    // ========================================================================
    // 3) Overflow/Underflow for Square
    // ========================================================================
    
    std::cout << "Testing overflow/underflow...\n";
    {
        std::vector<double> x = {-1e200, -1e100, -1e10, 1e10, 1e100, 1e200};
        for (auto dt: dtypes) {
            auto ref = ref_unary(x, "square");
            auto t0 = std::chrono::high_resolution_clock::now();
            Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
            Tensor y = square(a, 0);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
            report.add({ "overflow/square(" + get_dtype_name(dt) + ")", ok, msg,
                std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }
    
    // ========================================================================
    // 4) Very Small Values (Subnormals)
    // ========================================================================
    
    std::cout << "Testing subnormal values...\n";
    {
        double tiny = std::numeric_limits<double>::denorm_min();
        std::vector<double> x = { tiny, 2*tiny, -tiny, -2*tiny, 1e-300, -1e-300 };
        for (auto dt: dtypes) {
            for (auto& op: ops) {
                auto ref = ref_unary(x, op);
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
                Tensor y = apply_unary(a, op);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
                report.add({ "subnormals/" + op + "(" + get_dtype_name(dt) + ")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }
    
    // ========================================================================
    // 5) Integer Operations
    // ========================================================================
    
    std::cout << "Testing integer operations...\n";
    {
        std::vector<double> x = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
        for (Dtype idt: {Dtype::Int16, Dtype::Int32, Dtype::Int64}) {
            // Integer operations that should work
            for (std::string op: {"negate", "abs", "sign"}) {
                auto ref = ref_unary(x, op);
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, idt, {(int64_t)x.size()});
                Tensor y = apply_unary(a, op);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(idt), msg);
                report.add({ "integer/" + op + "(" + get_dtype_name(idt) + ")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
            
            // Square on integers (should promote to float64)
            {
                auto ref = ref_unary(x, "square");
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, idt, {(int64_t)x.size()});
                Tensor y = square(a, 0);  // Should output Float64
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = (y.dtype() == Dtype::Float64);
                if (ok) ok = check_tensor(y, ref, tol_for(Dtype::Float64), msg);
                else msg = "Expected Float64 output, got " + get_dtype_name(y.dtype());
                report.add({ "integer_promotion/square(" + get_dtype_name(idt) + ")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }
    
    // ========================================================================
    // 6) In-place on Integer Rejection
    // ========================================================================
    
    std::cout << "Testing in-place integer rejection...\n";
    {
        for (Dtype idt: {Dtype::Int16, Dtype::Int32, Dtype::Int64}) {
            for (std::string op: {"sqrt", "reciprocal"}) {
                Tensor a({{2}}, idt, DeviceIndex(Device::CPU), false);
                bool threw = false;
                auto t0 = std::chrono::high_resolution_clock::now();
                try { 
                    apply_unary_inplace(a, op); 
                } catch(const std::exception& e) { 
                    threw = true; 
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                report.add({ "integer_inplace_reject/" + op + "_(" + get_dtype_name(idt) + ")", 
                    threw, threw ? "Correctly threw exception" : "ERROR: Expected exception",
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }
    
    // ========================================================================
    // 7) Sign Function Edge Cases
    // ========================================================================
    
    std::cout << "Testing sign edge cases...\n";
    {
        std::vector<double> x = {-100.0, -1.0, -0.0, +0.0, 1.0, 100.0};
        for (auto dt: dtypes) {
            auto ref = ref_unary(x, "sign");
            auto t0 = std::chrono::high_resolution_clock::now();
            Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
            Tensor y = sign(a, 0);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
            report.add({ "sign/zero_handling(" + get_dtype_name(dt) + ")", ok, msg,
                std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }
    
    // ========================================================================
    // 8) Absolute Value on Mixed Signs
    // ========================================================================
    
    std::cout << "Testing abs on mixed signs...\n";
    {
        std::vector<double> x = {-1e10, -100.0, -1.0, -0.0, +0.0, 1.0, 100.0, 1e10};
        for (auto dt: dtypes) {
            auto ref = ref_unary(x, "abs");
            auto t0 = std::chrono::high_resolution_clock::now();
            Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
            Tensor y = abs(a, 0);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
            report.add({ "abs/mixed_signs(" + get_dtype_name(dt) + ")", ok, msg,
                std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }
    
    std::cout << "\nGenerating report...\n";
    report.write();
    
    std::cout << "\n✅ Edge cases report written to arith_edge_report.md\n";
    std::cout << "\n========================================\n";
    std::cout << " EDGE CASES TESTING COMPLETED\n";
    std::cout << "========================================\n\n";
    
    return 0;
}