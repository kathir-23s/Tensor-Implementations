#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ops/UnaryOps/Trigonometry.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>

using namespace OwnTensor;

int main() {
#ifndef WITH_CUDA
    std::cerr << "This test requires CUDA. Build with -DWITH_CUDA and nvcc.\n";
    return 1;
#else
    try {
        const Shape shape{{2, 4}};
        const DeviceIndex dev(Device::CUDA);
        const std::vector<float> base = { -3.14f, -1.57f, -0.5f, 0.0f, 0.5f, 1.57f, 3.14f, 4.71f };

        // ============================================================
        // Float32
        // ============================================================
        std::cout << "\n==== Float32 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Float32, dev, false);
            std::vector<float> v(base.begin(), base.end());
            a.set_data(v);

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 4);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nasin(a):\n";  asin(a).to_cpu().display(std::cout, 4);
            std::cout << "\nacos(a):\n";  acos(a).to_cpu().display(std::cout, 4);
            std::cout << "\natan(a):\n";  atan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 4);
            std::cout << "\nasinh(a):\n"; asinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\nacosh(a):\n"; acosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\natanh(a):\n"; atanh(a).to_cpu().display(std::cout, 4);

            Tensor t = a;
            sin_(t);   std::cout << "\nsin_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; cos_(t);   std::cout << "\ncos_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; tan_(t);   std::cout << "\ntan_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; asin_(t);  std::cout << "\nasin_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; acos_(t);  std::cout << "\nacos_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; atan_(t);  std::cout << "\natan_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; sinh_(t);  std::cout << "\nsinh_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; cosh_(t);  std::cout << "\ncosh_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; tanh_(t);  std::cout << "\ntanh_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; asinh_(t); std::cout << "\nasinh_(a):\n"; t.to_cpu().display(std::cout, 4);
            t = a; acosh_(t); std::cout << "\nacosh_(a):\n"; t.to_cpu().display(std::cout, 4);
            t = a; atanh_(t); std::cout << "\natanh_(a):\n"; t.to_cpu().display(std::cout, 4);
        }

        // ============================================================
        // Float64
        // ============================================================
        std::cout << "\n==== Float64 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Float64, dev, false);
            std::vector<double> v(base.begin(), base.end());
            a.set_data(v);

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 8);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 8);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 8);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 8);
            std::cout << "\nasin(a):\n";  asin(a).to_cpu().display(std::cout, 8);
            std::cout << "\nacos(a):\n";  acos(a).to_cpu().display(std::cout, 8);
            std::cout << "\natan(a):\n";  atan(a).to_cpu().display(std::cout, 8);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 8);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 8);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 8);
            std::cout << "\nasinh(a):\n"; asinh(a).to_cpu().display(std::cout, 8);
            std::cout << "\nacosh(a):\n"; acosh(a).to_cpu().display(std::cout, 8);
            std::cout << "\natanh(a):\n"; atanh(a).to_cpu().display(std::cout, 8);

            Tensor t = a;
            sin_(t);   std::cout << "\nsin_(a):\n";   t.to_cpu().display(std::cout, 8);
            t = a; cos_(t);   std::cout << "\ncos_(a):\n";   t.to_cpu().display(std::cout, 8);
            t = a; tan_(t);   std::cout << "\ntan_(a):\n";   t.to_cpu().display(std::cout, 8);
            t = a; sinh_(t);  std::cout << "\nsinh_(a):\n";  t.to_cpu().display(std::cout, 8);
            t = a; tanh_(t);  std::cout << "\ntanh_(a):\n";  t.to_cpu().display(std::cout, 8);
        }

        // ============================================================
        // Float16
        // ============================================================
        std::cout << "\n==== Float16 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Float16, dev, false);
            std::vector<float16_t> v(base.size());
            for (size_t i=0;i<base.size();++i) v[i] = float16_t(base[i]);
            a.set_data(v.data(), v.size());

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 4);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nasin(a):\n";  asin(a).to_cpu().display(std::cout, 4);
            std::cout << "\nacos(a):\n";  acos(a).to_cpu().display(std::cout, 4);
            std::cout << "\natan(a):\n";  atan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 4);
            std::cout << "\nasinh(a):\n"; asinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\nacosh(a):\n"; acosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\natanh(a):\n"; atanh(a).to_cpu().display(std::cout, 4);

            Tensor t = a;
            sin_(t);   std::cout << "\nsin_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; cos_(t);   std::cout << "\ncos_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; tan_(t);   std::cout << "\ntan_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; sinh_(t);  std::cout << "\nsinh_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; tanh_(t);  std::cout << "\ntanh_(a):\n";  t.to_cpu().display(std::cout, 4);
        }

        // ============================================================
        // Bfloat16
        // ============================================================
        std::cout << "\n==== Bfloat16 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Bfloat16, dev, false);
            std::vector<bfloat16_t> v(base.size());
            for (size_t i=0;i<base.size();++i) v[i] = bfloat16_t(base[i]);
            a.set_data(v.data(), v.size());

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 4);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nasin(a):\n";  asin(a).to_cpu().display(std::cout, 4);
            std::cout << "\nacos(a):\n";  acos(a).to_cpu().display(std::cout, 4);
            std::cout << "\natan(a):\n";  atan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 4);
            std::cout << "\nasinh(a):\n"; asinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\nacosh(a):\n"; acosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\natanh(a):\n"; atanh(a).to_cpu().display(std::cout, 4);

            Tensor t = a;
            sin_(t);   std::cout << "\nsin_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; cos_(t);   std::cout << "\ncos_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; tan_(t);   std::cout << "\ntan_(a):\n";   t.to_cpu().display(std::cout, 4);
            t = a; sinh_(t);  std::cout << "\nsinh_(a):\n";  t.to_cpu().display(std::cout, 4);
            t = a; tanh_(t);  std::cout << "\ntanh_(a):\n";  t.to_cpu().display(std::cout, 4);
        }

        // ============================================================
        // Int16
        // ============================================================
        std::cout << "\n==== Int16 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Int16, dev, false);
            std::vector<int16_t> v(base.size());
            for (size_t i=0;i<base.size();++i) v[i] = static_cast<int16_t>(base[i]);
            a.set_data(v);

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 4);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 4);
        }

        // ============================================================
        // Int32
        // ============================================================
        std::cout << "\n==== Int32 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Int32, dev, false);
            std::vector<int32_t> v(base.size());
            for (size_t i=0;i<base.size();++i) v[i] = static_cast<int32_t>(base[i]);
            a.set_data(v);

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 4);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 4);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 4);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 4);
        }

        // ============================================================
        // Int64
        // ============================================================
        std::cout << "\n==== Int64 (CUDA) ====\n";
        {
            Tensor a(shape, Dtype::Int64, dev, false);
            std::vector<int64_t> v(base.size());
            for (size_t i=0;i<base.size();++i) v[i] = static_cast<int64_t>(base[i]);
            a.set_data(v);

            std::cout << "a:\n"; a.to_cpu().display(std::cout, 8);

            std::cout << "\nsin(a):\n";   sin(a).to_cpu().display(std::cout, 8);
            std::cout << "\ncos(a):\n";   cos(a).to_cpu().display(std::cout, 8);
            std::cout << "\ntan(a):\n";   tan(a).to_cpu().display(std::cout, 8);
            std::cout << "\nsinh(a):\n";  sinh(a).to_cpu().display(std::cout, 8);
            std::cout << "\ncosh(a):\n";  cosh(a).to_cpu().display(std::cout, 8);
            std::cout << "\ntanh(a):\n";  tanh(a).to_cpu().display(std::cout, 8);
        }

        std::cout << "\n[Trig_test_naive_cuda_full complete]\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "CUDA Test Error: " << e.what() << "\n";
        return 1;

#endif // TRIG_TEST_CUDA_FULL_H
    }
}