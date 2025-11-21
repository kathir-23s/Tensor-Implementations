#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

bool is_close(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "Running ReLU Verification..." << std::endl;

    // Test 1: Basic values
    {
        std::cout << "Test 1: Basic values (-1, 0, 1)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}});
        t.set_data({-1.0f, 0.0f, 1.0f});
        
        Tensor out = mlp::ReLU(t);
        out.display();

        const float* data = out.data<float>();
        // ReLU(-1) = 0
        assert(is_close(data[0], 0.0f));
        // ReLU(0) = 0
        assert(is_close(data[1], 0.0f));
        // ReLU(1) = 1
        assert(is_close(data[2], 1.0f));
    }

    // Test 2: Large values (no saturation)
    {
        std::cout << "\nTest 2: Large values (no saturation)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2}});
        t.set_data({-10.0f, 10.0f});

        Tensor out = mlp::ReLU(t);
        out.display();
        
        const float* data = out.data<float>();
        assert(is_close(data[0], 0.0f));
        assert(is_close(data[1], 10.0f));
    }

    // Test 3: 2D Tensor Shape
    {
        std::cout << "\nTest 3: 2D Tensor Shape" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2, 2}});
        t.set_data({-2.0f, -1.0f, 1.0f, 2.0f});

        Tensor out = mlp::ReLU(t);
        
        assert(out.shape() == t.shape());
        const float* data = out.data<float>();
        assert(is_close(data[0], 0.0f));
        assert(is_close(data[1], 0.0f));
        assert(is_close(data[2], 1.0f));
        assert(is_close(data[3], 2.0f));
    }

    // Test 4: All negative values
    {
        std::cout << "\nTest 4: All negative values" << std::endl;
        Tensor t = Tensor::zeros(Shape{{4}});
        t.set_data({-5.0f, -3.0f, -1.0f, -0.1f});

        Tensor out = mlp::ReLU(t);
        
        const float* data = out.data<float>();
        for(int i=0; i<4; ++i) {
            assert(is_close(data[i], 0.0f));
        }
    }

    std::cout << "\n✅ ReLU Verification Passed!" << std::endl;
    return 0;
}
