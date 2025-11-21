#include "TensorLib.h"
#include "ops/UnaryOps/Reduction.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

bool is_close(float a, float b, float tol = 1e-5) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "Running Softmax Verification..." << std::endl;

    // Test 1: 1D Tensor
    {
        std::cout << "Test 1: 1D Tensor" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}});
        t.set_data({1.0f, 2.0f, 3.0f});
        
        Tensor out = mlp::softmax(t, 0);
        out.display();

        // Verify sum is 1
        Tensor sum_out = reduce_sum(out, {0});
        float sum_val = *sum_out.data<float>();
        std::cout << "Sum: " << sum_val << std::endl;
        assert(is_close(sum_val, 1.0f));
        
        // Verify values
        // exp(1)=2.718, exp(2)=7.389, exp(3)=20.085. Sum=30.192
        // out[0] = 0.090, out[1] = 0.244, out[2] = 0.665
        const float* data = out.data<float>();
        assert(is_close(data[0], 0.09003f));
        assert(is_close(data[1], 0.24473f));
        assert(is_close(data[2], 0.66524f));
    }

    // Test 2: 2D Tensor
    {
        std::cout << "\nTest 2: 2D Tensor (dim=1)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2, 2}});
        t.set_data({0.0f, 0.0f, 1.0f, 1.0f});

        Tensor out = mlp::softmax(t, 1);
        out.display();

        // Verify sum along dim 1 is [1, 1]
        Tensor sum_out = reduce_sum(out, {1});
        sum_out.display();
        
        const float* sum_data = sum_out.data<float>();
        assert(is_close(sum_data[0], 1.0f));
        assert(is_close(sum_data[1], 1.0f));
        
        // Row 0: softmax([0,0]) -> [0.5, 0.5]
        // Row 1: softmax([1,1]) -> [0.5, 0.5]
        const float* data = out.data<float>();
        assert(is_close(data[0], 0.5f));
        assert(is_close(data[1], 0.5f));
        assert(is_close(data[2], 0.5f));
        assert(is_close(data[3], 0.5f));
    }

    // Test 3: Numerical Stability
    {
        std::cout << "\nTest 3: Numerical Stability (Large values)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}});
        t.set_data({1000.0f, 1001.0f, 1002.0f});
        
        Tensor out = mlp::softmax(t, 0);
        
        // Verify sum is still 1
        Tensor sum_out = reduce_sum(out, {0});
        float sum_val = *sum_out.data<float>();
        assert(is_close(sum_val, 1.0f));
        
        // All values should be valid (not NaN or Inf)
        const float* data = out.data<float>();
        for(int i=0; i<3; ++i) {
            assert(!std::isnan(data[i]));
            assert(!std::isinf(data[i]));
        }
    }

    std::cout << "\n✅ Softmax Verification Passed!" << std::endl;
    return 0;
}
