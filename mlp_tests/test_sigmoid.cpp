#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

bool is_close(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "Running Sigmoid Verification..." << std::endl;

    // Test 1: Basic values
    {
        std::cout << "Test 1: Basic values (-1, 0, 1)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}});
        t.set_data({-1.0f, 0.0f, 1.0f});
        
        Tensor out = mlp::sigmoid(t);
        out.display();

        const float* data = out.data<float>();
        // sigmoid(-1) ≈ 0.26894
        assert(is_close(data[0], 0.26894f));
        // sigmoid(0) = 0.5
        assert(is_close(data[1], 0.5f));
        // sigmoid(1) ≈ 0.73105
        assert(is_close(data[2], 0.73105f));
    }

    // Test 2: Saturation
    {
        std::cout << "\nTest 2: Saturation (Large values)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2}});
        t.set_data({-10.0f, 10.0f});

        Tensor out = mlp::sigmoid(t);
        out.display();
        
        const float* data = out.data<float>();
        assert(is_close(data[0], 0.0f));
        assert(is_close(data[1], 1.0f));
    }

    // Test 3: 2D Tensor Shape
    {
        std::cout << "\nTest 3: 2D Tensor Shape" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2, 2}});
        t.set_data({0.0f, 0.0f, 0.0f, 0.0f});

        Tensor out = mlp::sigmoid(t);
        
        assert(out.shape() == t.shape());
        const float* data = out.data<float>();
        for(int i=0; i<4; ++i) {
            assert(is_close(data[i], 0.5f));
        }
    }

    std::cout << "\n✅ Sigmoid Verification Passed!" << std::endl;
    return 0;
}
