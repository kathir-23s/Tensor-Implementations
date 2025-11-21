#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

bool is_close(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "Running Tanh Verification..." << std::endl;

    // Test 1: Basic values
    {
        std::cout << "Test 1: Basic values (-1, 0, 1)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}});
        t.set_data({-1.0f, 0.0f, 1.0f});
        
        Tensor out = mlp::tanh(t);
        out.display();

        const float* data = out.data<float>();
        // tanh(-1) ≈ -0.76159
        assert(is_close(data[0], -0.76159f));
        // tanh(0) = 0
        assert(is_close(data[1], 0.0f));
        // tanh(1) ≈ 0.76159
        assert(is_close(data[2], 0.76159f));
    }

    // Test 2: Saturation
    {
        std::cout << "\nTest 2: Saturation (Large values)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2}});
        t.set_data({-10.0f, 10.0f});

        Tensor out = mlp::tanh(t);
        out.display();
        
        const float* data = out.data<float>();
        assert(is_close(data[0], -1.0f));
        assert(is_close(data[1], 1.0f));
    }

    // Test 3: 2D Tensor Shape
    {
        std::cout << "\nTest 3: 2D Tensor Shape" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2, 2}});
        t.set_data({-0.5f, 0.5f, -0.5f, 0.5f});

        Tensor out = mlp::tanh(t);
        
        assert(out.shape() == t.shape());
        const float* data = out.data<float>();
        // tanh(-0.5) ≈ -0.46211
        // tanh(0.5) ≈ 0.46211
        assert(is_close(data[0], -0.46211f));
        assert(is_close(data[1], 0.46211f));
        assert(is_close(data[2], -0.46211f));
        assert(is_close(data[3], 0.46211f));
    }

    std::cout << "\n✅ Tanh Verification Passed!" << std::endl;
    return 0;
}
