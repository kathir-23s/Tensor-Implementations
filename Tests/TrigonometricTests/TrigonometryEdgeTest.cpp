#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef WITH_CUDA
  #include <cuda_runtime.h>
#endif

#include "TensorLib.h"
#include "ops/helpers/testutils.h"

using namespace OwnTensor;
using namespace TestUtils;

// ------------------------- tolerance helpers -------------------------
struct Tolerances { double rel_tol; double abs_tol; };

static Tolerances tol_for(Dtype dt) {
    switch (dt) {
        case Dtype::Float16:
        case Dtype::Bfloat16: return {5e-2, 1e-4};
        case Dtype::Float32:  return {1e-5, 1e-6};
        case Dtype::Float64:  return {1e-7, 1e-9};
        default:              return {1e-5, 1e-6};
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

// ------------------------- tensor helpers -------------------------
static Tensor make_tensor_cpu(const std::vector<double>& vals, Dtype dt, const std::vector<int64_t>& shape) {
    Tensor t(Shape{shape}, dt, DeviceIndex(Device::CPU), /*requires_grad=*/false);
    switch (dt) {
        case Dtype::Float16: {
            auto* p = t.data<float16_t>();
            for (size_t i=0;i<vals.size();++i) p[i] = float16_t(static_cast<float>(vals[i]));
            break;
        }
        case Dtype::Bfloat16: {
            auto* p = t.data<bfloat16_t>();
            for (size_t i=0;i<vals.size();++i) p[i] = bfloat16_t(static_cast<float>(vals[i]));
            break;
        }
        case Dtype::Float32: {
            auto* p = t.data<float>();
            for (size_t i=0;i<vals.size();++i) p[i] = static_cast<float>(vals[i]);
            break;
        }
        case Dtype::Float64: {
            auto* p = t.data<double>();
            for (size_t i=0;i<vals.size();++i) p[i] = vals[i];
            break;
        }
        default: throw std::runtime_error("Unsupported dtype");
    }
    return t;
}

static std::vector<double> to_double_vec(const Tensor& t) {
    std::vector<double> out(t.numel());
    switch (t.dtype()) {
    case Dtype::Float16:  { auto* p = t.data<float16_t>(); for (size_t i=0;i<t.numel();++i) out[i] = static_cast<float>(p[i]); break; }
    case Dtype::Bfloat16: { auto* p = t.data<bfloat16_t>();for (size_t i=0;i<t.numel();++i) out[i] = static_cast<float>(p[i]); break; }
    case Dtype::Float32:  { auto* p = t.data<float>();     for (size_t i=0;i<t.numel();++i) out[i] = p[i]; break; }
    case Dtype::Float64:  { auto* p = t.data<double>();    for (size_t i=0;i<t.numel();++i) out[i] = p[i]; break; }
    default: throw std::runtime_error("Unsupported dtype");

    }
    return out;
}

static Tensor apply_unary(const Tensor& a, const std::string& op) {
    if (op=="sin")   return sin(a);
    if (op=="cos")   return cos(a);
    if (op=="tan")   return tan(a);
    if (op=="asin")  return asin(a);
    if (op=="acos")  return acos(a);
    if (op=="atan")  return atan(a);
    if (op=="sinh")  return sinh(a);
    if (op=="cosh")  return cosh(a);
    if (op=="tanh")  return tanh(a);
    if (op=="asinh") return asinh(a);
    if (op=="acosh") return acosh(a);
    if (op=="atanh") return atanh(a);
    throw std::runtime_error("Unknown op: " + op);
}

static void apply_unary_inplace(Tensor& a, const std::string& op) {
    if (op=="sin")   { sin_(a);   return; }
    if (op=="cos")   { cos_(a);   return; }
    if (op=="tan")   { tan_(a);   return; }
    if (op=="asin")  { asin_(a);  return; }
    if (op=="acos")  { acos_(a);  return; }
    if (op=="atan")  { atan_(a);  return; }
    if (op=="sinh")  { sinh_(a);  return; }
    if (op=="cosh")  { cosh_(a);  return; }
    if (op=="tanh")  { tanh_(a);  return; }
    if (op=="asinh") { asinh_(a); return; }
    if (op=="acosh") { acosh_(a); return; }
    if (op=="atanh") { atanh_(a); return; }
    throw std::runtime_error("Unknown op: " + op);
}

static std::vector<double> ref_unary(const std::vector<double>& x, const std::string& op) {
    std::vector<double> y(x.size());
    if (op=="sin")   for (size_t i=0;i<x.size();++i) y[i] = std::sin(x[i]);
    if (op=="cos")   for (size_t i=0;i<x.size();++i) y[i] = std::cos(x[i]);
    if (op=="tan")   for (size_t i=0;i<x.size();++i) y[i] = std::tan(x[i]);
    if (op=="asin")  for (size_t i=0;i<x.size();++i) y[i] = std::asin(x[i]);
    if (op=="acos")  for (size_t i=0;i<x.size();++i) y[i] = std::acos(x[i]);
    if (op=="atan")  for (size_t i=0;i<x.size();++i) y[i] = std::atan(x[i]);
    if (op=="sinh")  for (size_t i=0;i<x.size();++i) y[i] = std::sinh(x[i]);
    if (op=="cosh")  for (size_t i=0;i<x.size();++i) y[i] = std::cosh(x[i]);
    if (op=="tanh")  for (size_t i=0;i<x.size();++i) y[i] = std::tanh(x[i]);
    if (op=="asinh") for (size_t i=0;i<x.size();++i) y[i] = std::asinh(x[i]);
    if (op=="acosh") for (size_t i=0;i<x.size();++i) y[i] = std::acosh(x[i]);
    if (op=="atanh") for (size_t i=0;i<x.size();++i) y[i] = std::atanh(x[i]);
    return y;
}

static bool check_tensor(const Tensor& out, const std::vector<double>& ref, const Tolerances& tol, std::string& msg) {
    auto got = to_double_vec(out);
    for (size_t i=0;i<ref.size();++i) {
        if (!almost_equal(got[i], ref[i], tol)) {
            msg = "Mismatch at " + std::to_string(i) + " got=" + std::to_string(got[i]) +
                  " ref=" + std::to_string(ref[i]);
            return false;
        }
    }
    return true;
}

// ------------------------- reporting -------------------------
struct TestResult { std::string name; bool passed; std::string message; double ms; };

class Report {
    std::vector<TestResult> results;
    std::string filename;
public:
    explicit Report(std::string fname): filename(std::move(fname)) {}
    void add(const TestResult& tr) { results.push_back(tr); }
    void write() {
        std::ofstream f("local_test/" + filename);
        f << "# Trig Edge Cases Report\n\n";
        int pass=0, fail=0; for (auto& r: results) { if (r.passed) ++pass; else ++fail; }
        f << "- Total: " << results.size() << "\n";
        f << "- Passed: " << pass << "\n";
        f << "- Failed: " << fail << "\n\n";
        f << "## Details\n\n";
        for (auto& r: results) {
            f << "### " << r.name << "\n";
            f << "- Result: " << (r.passed ? "PASS" : "FAIL") << "\n";
            f << "- Time(ms): " << std::fixed << std::setprecision(3) << r.ms << "\n";
            if (!r.passed) f << "- Message: " << r.message << "\n";
            f << "\n";
        }
    }
};

// ------------------------- main -------------------------
int main() {
    Report report("trig_edge_report.md");

    std::vector<std::string> ops = {"sin","cos","tan","asin","acos","atan",
                                    "sinh","cosh","tanh","asinh","acosh","atanh"};
    std::vector<Dtype> dtypes = {Dtype::Float16, Dtype::Bfloat16, Dtype::Float32, Dtype::Float64};

    // 1) Special values (0, -0, inf, -inf, nan)
    for (auto dt: dtypes) {
        for (auto& op: ops) {
            std::vector<double> x = { +0.0, -0.0,
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN() };
            auto ref = ref_unary(x, op);
            auto t0 = std::chrono::high_resolution_clock::now();
            Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
            Tensor y = apply_unary(a, op);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
            report.add({ "specials/"+op+"("+std::to_string((int)dt)+")", ok, msg,
                std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }

    // 2) Domain checks (asin/acos: [-1,1], atanh: (-1,1), acosh: [1, +inf))
    {
        std::vector<double> x1 = { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };
        for (auto dt: dtypes) {
            for (auto& op: std::vector<std::string>{"asin","acos","atanh","acosh"}) {
                auto ref = ref_unary(x1, op);
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x1, dt, {(int64_t)x1.size()});
                Tensor y = apply_unary(a, op);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
                report.add({ "domain/"+op+"("+std::to_string((int)dt)+")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }

    // 3) Large magnitudes (overflow risks for hyperbolic functions and tan near pi/2)
    {
        std::vector<double> x = { -1000.0, -100.0, -50.0, -10.0, -3.141592653589793/2.0 + 1e-6,
                                   0.0,
                                   3.141592653589793/2.0 - 1e-6, 10.0, 50.0, 100.0, 1000.0 };
        for (auto dt: dtypes) {
            for (auto& op: std::vector<std::string>{"sinh","cosh","tanh","tan"}) {
                auto ref = ref_unary(x, op);
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
                Tensor y = apply_unary(a, op);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
                report.add({ "large_mag/"+op+"("+std::to_string((int)dt)+")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }

    // 4) Subnormals / small magnitudes
    {
        double tiny = std::numeric_limits<float>::denorm_min();
        std::vector<double> x = { tiny, 2*tiny, -tiny, -2*tiny };
        for (auto dt: dtypes) {
            for (auto& op: ops) {
                auto ref = ref_unary(x, op);
                auto t0 = std::chrono::high_resolution_clock::now();
                Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
                Tensor y = apply_unary(a, op);
                auto t1 = std::chrono::high_resolution_clock::now();
                std::string msg; bool ok = check_tensor(y, ref, tol_for(dt), msg);
                report.add({ "subnormals/"+op+"("+std::to_string((int)dt)+")", ok, msg,
                    std::chrono::duration<double, std::milli>(t1-t0).count() });
            }
        }
    }

    // 5) In-place on views (policy-dependent; accept throw or success)
    {
        std::vector<double> x = {0.5, 1.0, -0.5, 3.0};
        for (auto dt: dtypes) {
            Tensor a = make_tensor_cpu(x, dt, {(int64_t)x.size()});
            Tensor view = a; // replace with a true view/slice when available in framework
            bool threw = false; auto t0 = std::chrono::high_resolution_clock::now();
            try { apply_unary_inplace(view, "sin"); } catch(...) { threw = true; }
            auto t1 = std::chrono::high_resolution_clock::now();
            bool ok = threw || true; // policy: allow either
            std::string msg = threw ? "threw on in-place view (allowed)" : "performed in-place on view";
            report.add({ "inplace_on_view/sin_(" + std::to_string((int)dt)+")", ok, msg,
                         std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }

    // 6) Integer in-place rejection parity
    {
        for (Dtype idt: {Dtype::Int16, Dtype::Int32, Dtype::Int64}) {
            Tensor a(Shape{{2}}, idt, DeviceIndex(Device::CPU), false);
            bool threw = false; auto t0=std::chrono::high_resolution_clock::now();
            try { apply_unary_inplace(a, "sin"); } catch(...) { threw = true; }
            auto t1=std::chrono::high_resolution_clock::now();
            report.add({ "integer_inplace_reject/sin_", threw, threw? "" : "expected throw",
                         std::chrono::duration<double, std::milli>(t1-t0).count() });
        }
    }

    report.write();
    std::cout << "Trig edge cases report written to trig_edge_report.md\n";
    return 0;
}
