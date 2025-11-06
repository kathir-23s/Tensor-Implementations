#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ops/UnaryOps/Trigonometry.h"   // sin, cos, tanh, ... and in-place versions

#include <iostream>
#include <vector>
#include <cstdint>

using namespace OwnTensor;

static void print_header(const char* title) {
    std::cout << "\n==== " << title << " ====\n";
}

[[maybe_unused]] static int precision_for(Dtype dt) {
  return dt == Dtype::Float64 ? 10 : 6;
}

int main() {
    try {
        const Shape shape{{2, 4}};
        // Common float host values (integers will truncate)
        const std::vector<float> base = { -3.14f, -1.57f, -0.5f, 0.0f, 0.5f, 1.57f, 3.14f, 4.71f };

        // ============================================================
        // Float32
        // ============================================================
        {
            print_header("Float32");
            Tensor a(shape, Dtype::Float32, DeviceIndex(Device::CPU), false);
            {
                std::vector<float> v = base; // exact dtype match
                a.set_data(v);
            }

            std::cout << "a (f32):\n"; a.display(std::cout, 6);

            Tensor s  = sin(a);
            Tensor c  = cos(a);
            Tensor ta = tan(a);
            Tensor as = asin(a);
            Tensor ac = acos(a);
            Tensor at = atan(a);
            Tensor sh = sinh(a);
            Tensor ch = cosh(a);
            Tensor th = tanh(a);
            Tensor ash = asinh(a);
            Tensor ach = acosh(a);
            Tensor ath = atanh(a);

            std::cout << "\nsin(a):\n";   s.display(std::cout, 6);
            std::cout << "\ncos(a):\n";   c.display(std::cout, 6);
            std::cout << "\ntan(a):\n";   ta.display(std::cout, 6);
            std::cout << "\nasin(a):\n";  as.display(std::cout, 6);
            std::cout << "\nacos(a):\n";  ac.display(std::cout, 6);
            std::cout << "\natan(a):\n";  at.display(std::cout, 6);
            std::cout << "\nsinh(a):\n";  sh.display(std::cout, 6);
            std::cout << "\ncosh(a):\n";  ch.display(std::cout, 6);
            std::cout << "\ntanh(a):\n";  th.display(std::cout, 6);
            std::cout << "\nasinh(a):\n"; ash.display(std::cout, 6);
            std::cout << "\nacosh(a):\n"; ach.display(std::cout, 6);
            std::cout << "\natanh(a):\n"; ath.display(std::cout, 6);

            // In-place (float dtypes only)
            {
                Tensor t = a; sin_(t);   std::cout << "\n[float32] sin_(a):\n";   t.display(std::cout, 6);
                t = a;        cos_(t);   std::cout << "\n[float32] cos_(a):\n";   t.display(std::cout, 6);
                t = a;        tan_(t);   std::cout << "\n[float32] tan_(a):\n";   t.display(std::cout, 6);
                t = a;        asin_(t);  std::cout << "\n[float32] asin_(a):\n";  t.display(std::cout, 6);
                t = a;        acos_(t);  std::cout << "\n[float32] acos_(a):\n";  t.display(std::cout, 6);
                t = a;        atan_(t);  std::cout << "\n[float32] atan_(a):\n";  t.display(std::cout, 6);
                t = a;        sinh_(t);  std::cout << "\n[float32] sinh_(a):\n";  t.display(std::cout, 6);
                t = a;        cosh_(t);  std::cout << "\n[float32] cosh_(a):\n";  t.display(std::cout, 6);
                t = a;        tanh_(t);  std::cout << "\n[float32] tanh_(a):\n";  t.display(std::cout, 6);
                t = a;        asinh_(t); std::cout << "\n[float32] asinh_(a):\n"; t.display(std::cout, 6);
                t = a;        acosh_(t); std::cout << "\n[float32] acosh_(a):\n"; t.display(std::cout, 6);
                t = a;        atanh_(t); std::cout << "\n[float32] atanh_(a):\n"; t.display(std::cout, 6);
            }
        }

        // ============================================================
        // Float64
        // ============================================================
        {
            print_header("Float64");
            Tensor a(shape, Dtype::Float64, DeviceIndex(Device::CPU), false);
            {
                std::vector<double> v(base.size());
                for (size_t i = 0; i < base.size(); ++i) v[i] = static_cast<double>(base[i]);
                a.set_data(v);
            }

            std::cout << "a (f64):\n"; a.display(std::cout, 10);

            Tensor s  = sin(a);
            Tensor c  = cos(a);
            Tensor ta = tan(a);
            Tensor as = asin(a);
            Tensor ac = acos(a);
            Tensor at = atan(a);
            Tensor sh = sinh(a);
            Tensor ch = cosh(a);
            Tensor th = tanh(a);
            Tensor ash = asinh(a);
            Tensor ach = acosh(a);
            Tensor ath = atanh(a);

            std::cout << "\nsin(a):\n";   s.display(std::cout, 10);
            std::cout << "\ncos(a):\n";   c.display(std::cout, 10);
            std::cout << "\ntan(a):\n";   ta.display(std::cout, 10);
            std::cout << "\nasin(a):\n";  as.display(std::cout, 10);
            std::cout << "\nacos(a):\n";  ac.display(std::cout, 10);
            std::cout << "\natan(a):\n";  at.display(std::cout, 10);
            std::cout << "\nsinh(a):\n";  sh.display(std::cout, 10);
            std::cout << "\ncosh(a):\n";  ch.display(std::cout, 10);
            std::cout << "\ntanh(a):\n";  th.display(std::cout, 10);
            std::cout << "\nasinh(a):\n"; ash.display(std::cout, 10);
            std::cout << "\nacosh(a):\n"; ach.display(std::cout, 10);
            std::cout << "\natanh(a):\n"; ath.display(std::cout, 10);

            // In-place (float dtypes only)
            {
                Tensor t = a; sin_(t);   std::cout << "\n[float64] sin_(a):\n";   t.display(std::cout, 10);
                t = a;        cos_(t);   std::cout << "\n[float64] cos_(a):\n";   t.display(std::cout, 10);
                t = a;        tan_(t);   std::cout << "\n[float64] tan_(a):\n";   t.display(std::cout, 10);
                t = a;        asin_(t);  std::cout << "\n[float64] asin_(a):\n";  t.display(std::cout, 10);
                t = a;        acos_(t);  std::cout << "\n[float64] acos_(a):\n";  t.display(std::cout, 10);
                t = a;        atan_(t);  std::cout << "\n[float64] atan_(a):\n";  t.display(std::cout, 10);
                t = a;        sinh_(t);  std::cout << "\n[float64] sinh_(a):\n";  t.display(std::cout, 10);
                t = a;        cosh_(t);  std::cout << "\n[float64] cosh_(a):\n";  t.display(std::cout, 10);
                t = a;        tanh_(t);  std::cout << "\n[float64] tanh_(a):\n";  t.display(std::cout, 10);
                t = a;        asinh_(t); std::cout << "\n[float64] asinh_(a):\n"; t.display(std::cout, 10);
                t = a;        acosh_(t); std::cout << "\n[float64] acosh_(a):\n"; t.display(std::cout, 10);
                t = a;        atanh_(t); std::cout << "\n[float64] atanh_(a):\n"; t.display(std::cout, 10);
            }
        }

        // ============================================================
        // Float16
        // ============================================================
        {
            print_header("Float16");
            Tensor a(shape, Dtype::Float16, DeviceIndex(Device::CPU), false);
            {
                std::vector<float16_t> v(base.size());
                for (size_t i = 0; i < base.size(); ++i) v[i] = float16_t(base[i]); // explicit elementwise convert
                a.set_data(v.data(), v.size()); // use bytes overload if your API expects bytes
            }

            std::cout << "a (f16):\n"; a.display(std::cout, 6);

            Tensor s  = sin(a);
            Tensor c  = cos(a);
            Tensor ta = tan(a);
            Tensor as = asin(a);
            Tensor ac = acos(a);
            Tensor at = atan(a);
            Tensor sh = sinh(a);
            Tensor ch = cosh(a);
            Tensor th = tanh(a);
            Tensor ash = asinh(a);
            Tensor ach = acosh(a);
            Tensor ath = atanh(a);

            std::cout << "\nsin(a):\n";   s.display(std::cout, 6);
            std::cout << "\ncos(a):\n";   c.display(std::cout, 6);
            std::cout << "\ntan(a):\n";   ta.display(std::cout, 6);
            std::cout << "\nasin(a):\n";  as.display(std::cout, 6);
            std::cout << "\nacos(a):\n";  ac.display(std::cout, 6);
            std::cout << "\natan(a):\n";  at.display(std::cout, 6);
            std::cout << "\nsinh(a):\n";  sh.display(std::cout, 6);
            std::cout << "\ncosh(a):\n";  ch.display(std::cout, 6);
            std::cout << "\ntanh(a):\n";  th.display(std::cout, 6);
            std::cout << "\nasinh(a):\n"; ash.display(std::cout, 6);
            std::cout << "\nacosh(a):\n"; ach.display(std::cout, 6);
            std::cout << "\natanh(a):\n"; ath.display(std::cout, 6);

            // In-place
            {
                Tensor t = a; sin_(t);   std::cout << "\n[float16] sin_(a):\n";   t.display(std::cout, 6);
                t = a;        cos_(t);   std::cout << "\n[float16] cos_(a):\n";   t.display(std::cout, 6);
                t = a;        tan_(t);   std::cout << "\n[float16] tan_(a):\n";   t.display(std::cout, 6);
                t = a;        asin_(t);  std::cout << "\n[float16] asin_(a):\n";  t.display(std::cout, 6);
                t = a;        acos_(t);  std::cout << "\n[float16] acos_(a):\n";  t.display(std::cout, 6);
                t = a;        atan_(t);  std::cout << "\n[float16] atan_(a):\n";  t.display(std::cout, 6);
                t = a;        sinh_(t);  std::cout << "\n[float16] sinh_(a):\n";  t.display(std::cout, 6);
                t = a;        cosh_(t);  std::cout << "\n[float16] cosh_(a):\n";  t.display(std::cout, 6);
                t = a;        tanh_(t);  std::cout << "\n[float16] tanh_(a):\n";  t.display(std::cout, 6);
                t = a;        asinh_(t); std::cout << "\n[float16] asinh_(a):\n"; t.display(std::cout, 6);
                t = a;        acosh_(t); std::cout << "\n[float16] acosh_(a):\n"; t.display(std::cout, 6);
                t = a;        atanh_(t); std::cout << "\n[float16] atanh_(a):\n"; t.display(std::cout, 6);
            }
        }

        // ============================================================
        // Bfloat16
        // ============================================================
        {
            print_header("Bfloat16");
            Tensor a(shape, Dtype::Bfloat16, DeviceIndex(Device::CPU), false);
            {
                std::vector<bfloat16_t> v(base.size());
                for (size_t i = 0; i < base.size(); ++i) v[i] = bfloat16_t(base[i]); // explicit elementwise convert
                a.set_data(v.data(), v.size());
            }

            std::cout << "a (bf16):\n"; a.display(std::cout, 6);

            Tensor s  = sin(a);
            Tensor c  = cos(a);
            Tensor ta = tan(a);
            Tensor as = asin(a);
            Tensor ac = acos(a);
            Tensor at = atan(a);
            Tensor sh = sinh(a);
            Tensor ch = cosh(a);
            Tensor th = tanh(a);
            Tensor ash = asinh(a);
            Tensor ach = acosh(a);
            Tensor ath = atanh(a);

            std::cout << "\nsin(a):\n";   s.display(std::cout, 6);
            std::cout << "\ncos(a):\n";   c.display(std::cout, 6);
            std::cout << "\ntan(a):\n";   ta.display(std::cout, 6);
            std::cout << "\nasin(a):\n";  as.display(std::cout, 6);
            std::cout << "\nacos(a):\n";  ac.display(std::cout, 6);
            std::cout << "\natan(a):\n";  at.display(std::cout, 6);
            std::cout << "\nsinh(a):\n";  sh.display(std::cout, 6);
            std::cout << "\ncosh(a):\n";  ch.display(std::cout, 6);
            std::cout << "\ntanh(a):\n";  th.display(std::cout, 6);
            std::cout << "\nasinh(a):\n"; ash.display(std::cout, 6);
            std::cout << "\nacosh(a):\n"; ach.display(std::cout, 6);
            std::cout << "\natanh(a):\n"; ath.display(std::cout, 6);

            // In-place
            {
                Tensor t = a; sin_(t);   std::cout << "\n[bf16] sin_(a):\n";   t.display(std::cout, 6);
                t = a;        cos_(t);   std::cout << "\n[bf16] cos_(a):\n";   t.display(std::cout, 6);
                t = a;        tan_(t);   std::cout << "\n[bf16] tan_(a):\n";   t.display(std::cout, 6);
                t = a;        asin_(t);  std::cout << "\n[bf16] asin_(a):\n";  t.display(std::cout, 6);
                t = a;        acos_(t);  std::cout << "\n[bf16] acos_(a):\n";  t.display(std::cout, 6);
                t = a;        atan_(t);  std::cout << "\n[bf16] atan_(a):\n";  t.display(std::cout, 6);
                t = a;        sinh_(t);  std::cout << "\n[bf16] sinh_(a):\n";  t.display(std::cout, 6);
                t = a;        cosh_(t);  std::cout << "\n[bf16] cosh_(a):\n";  t.display(std::cout, 6);
                t = a;        tanh_(t);  std::cout << "\n[bf16] tanh_(a):\n";  t.display(std::cout, 6);
                t = a;        asinh_(t); std::cout << "\n[bf16] asinh_(a):\n"; t.display(std::cout, 6);
                t = a;        acosh_(t); std::cout << "\n[bf16] acosh_(a):\n"; t.display(std::cout, 6);
                t = a;        atanh_(t); std::cout << "\n[bf16] atanh_(a):\n"; t.display(std::cout, 6);
            }
        }

        // ============================================================
        // Int16  (out-of-place only)
        // ============================================================
        {
            print_header("Int16");
            Tensor a(shape, Dtype::Int16, DeviceIndex(Device::CPU), false);
            {
                std::vector<int16_t> v(base.size());
                for (size_t i = 0; i < base.size(); ++i) v[i] = static_cast<int16_t>(base[i]);
                a.set_data(v);
            }

            std::cout << "a (i16):\n"; a.display(std::cout, 6);

            Tensor s  = sin(a);  std::cout << "\nsin(a):\n";  s.display(std::cout, 6);
            Tensor c  = cos(a);  std::cout << "\ncos(a):\n";  c.display(std::cout, 6);
            Tensor ta = tan(a);  std::cout << "\ntan(a):\n";  ta.display(std::cout, 6);
            Tensor as = asin(a); std::cout << "\nasin(a):\n"; as.display(std::cout, 6);
            Tensor ac = acos(a); std::cout << "\nacos(a):\n"; ac.display(std::cout, 6);
            Tensor at = atan(a); std::cout << "\natan(a):\n"; at.display(std::cout, 6);
            Tensor sh = sinh(a); std::cout << "\nsinh(a):\n"; sh.display(std::cout, 6);
            Tensor ch = cosh(a); std::cout << "\ncosh(a):\n"; ch.display(std::cout, 6);
            Tensor th = tanh(a); std::cout << "\ntanh(a):\n"; th.display(std::cout, 6);
            Tensor ash = asinh(a); std::cout << "\nasinh(a):\n"; ash.display(std::cout, 6);
            Tensor ach = acosh(a); std::cout << "\nacosh(a):\n"; ach.display(std::cout, 6);
            Tensor ath = atanh(a); std::cout << "\natanh(a):\n"; ath.display(std::cout, 6);
        }

        // ============================================================
        // Int32  (out-of-place only)
        // ============================================================
        {
            print_header("Int32");
            Tensor a(shape, Dtype::Int32, DeviceIndex(Device::CPU), false);
            {
                std::vector<int32_t> v(base.size());
                for (size_t i = 0; i < base.size(); ++i) v[i] = static_cast<int32_t>(base[i]);
                a.set_data(v);
            }

            std::cout << "a (i32):\n"; a.display(std::cout, 6);

            Tensor s  = sin(a);  std::cout << "\nsin(a):\n";  s.display(std::cout, 6);
            Tensor c  = cos(a);  std::cout << "\ncos(a):\n";  c.display(std::cout, 6);
            Tensor ta = tan(a);  std::cout << "\ntan(a):\n";  ta.display(std::cout, 6);
            Tensor as = asin(a); std::cout << "\nasin(a):\n"; as.display(std::cout, 6);
            Tensor ac = acos(a); std::cout << "\nacos(a):\n"; ac.display(std::cout, 6);
            Tensor at = atan(a); std::cout << "\natan(a):\n"; at.display(std::cout, 6);
            Tensor sh = sinh(a); std::cout << "\nsinh(a):\n"; sh.display(std::cout, 6);
            Tensor ch = cosh(a); std::cout << "\ncosh(a):\n"; ch.display(std::cout, 6);
            Tensor th = tanh(a); std::cout << "\ntanh(a):\n"; th.display(std::cout, 6);
            Tensor ash = asinh(a); std::cout << "\nasinh(a):\n"; ash.display(std::cout, 6);
            Tensor ach = acosh(a); std::cout << "\nacosh(a):\n"; ach.display(std::cout, 6);
            Tensor ath = atanh(a); std::cout << "\natanh(a):\n"; ath.display(std::cout, 6);
        }

        // ============================================================
        // Int64  (out-of-place only)
        // ============================================================
        {
            print_header("Int64");
            Tensor a(shape, Dtype::Int64, DeviceIndex(Device::CPU), false);
            {
                std::vector<int64_t> v(base.size());
                for (size_t i = 0; i < base.size(); ++i) v[i] = static_cast<int64_t>(base[i]);
                a.set_data(v);
            }

            std::cout << "a (i64):\n"; a.display(std::cout, 10);

            Tensor s  = sin(a);  std::cout << "\nsin(a):\n";  s.display(std::cout, 10);
            Tensor c  = cos(a);  std::cout << "\ncos(a):\n";  c.display(std::cout, 10);
            Tensor ta = tan(a);  std::cout << "\ntan(a):\n";  ta.display(std::cout, 10);
            Tensor as = asin(a); std::cout << "\nasin(a):\n"; as.display(std::cout, 10);
            Tensor ac = acos(a); std::cout << "\nacos(a):\n"; ac.display(std::cout, 10);
            Tensor at = atan(a); std::cout << "\natan(a):\n"; at.display(std::cout, 10);
            Tensor sh = sinh(a); std::cout << "\nsinh(a):\n"; sh.display(std::cout, 10);
            Tensor ch = cosh(a); std::cout << "\ncosh(a):\n"; ch.display(std::cout, 10);
            Tensor th = tanh(a); std::cout << "\ntanh(a):\n"; th.display(std::cout, 10);
            Tensor ash = asinh(a); std::cout << "\nasinh(a):\n"; ash.display(std::cout, 10);
            Tensor ach = acosh(a); std::cout << "\nacosh(a):\n"; ach.display(std::cout, 10);
            Tensor ath = atanh(a); std::cout << "\natanh(a):\n"; ath.display(std::cout, 10);
        }

        std::cout << "\nAll dtype tests completed successfully.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
