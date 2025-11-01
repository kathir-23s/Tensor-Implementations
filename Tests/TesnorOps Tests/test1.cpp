#include "test_common.h"

template<typename T>
Tensor make_tensor_from_values(const std::vector<T>& vals, const std::string& device, Dtype dtype) {
    Tensor t(Shape{(int64_t)vals.size()}, TensorOptions().with_dtype(dtype).with_device(device));
    t.set_data(vals);
    return t;
}

template<typename T>
void test_add_op(const std::string& device, Dtype dtype) {
    auto a_vals = get_test_data<T>();
    auto b_vals = get_test_data<T>();

    Tensor a = make_tensor_from_values(a_vals, device, dtype);
    Tensor b = make_tensor_from_values(b_vals, device, dtype);

    Tensor res = a + b;

    std::cout << "Addition result on device " << device << " dtype " << dtype_to_string(dtype) << ":\n";
    res.display(std::cout, 6);
    std::cout << std::endl;

    // Further checks can go here, e.g. element-wise compares to expected results
}


int main() {
    for (const auto& device : devices) {
        for (auto dtype : test_dtypes) {
            if (device == "cuda" ) continue;
            switch (dtype) {
                case Dtype::Int64:     test_add_op<int64_t>(device, dtype); break;
                case Dtype::Int32:     test_add_op<int32_t>(device, dtype); break;
                case Dtype::Float64:   test_add_op<double>(device, dtype); break;
                case Dtype::Float32:   test_add_op<float>(device, dtype); break;
                case Dtype::Float16:   test_add_op<float16_t>(device, dtype); break;
                case Dtype::Bfloat16:  test_add_op<bfloat16_t>(device, dtype); break;
                default: break;
            }
        }
    }
    return 0;
}
