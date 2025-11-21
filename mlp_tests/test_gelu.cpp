#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

bool is_close(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

// Reference GELU implementation using tanh approximation
float gelu_reference(float x) {
    float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    return 0.5f * x * (1.0f + std::tanh(tanh_arg));
}

int main() {
    std::cout << "Running GELU Verification..." << std::endl;

    TensorOptions options;
    options.device = Device::CUDA;
    options.dtype = Dtype::Float32;
    options.requires_grad = false;

    // Test 1: Basic values
    {
        std::cout << "Test 1: Basic values (-2, -1, 0, 1, 2)" << std::endl;
        Tensor t = Tensor::zeros(Shape{{5}}, options);
        t.set_data({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
        
        Tensor out = mlp::GeLU(t);
        out.to_cpu().display();

        const float* data = out.data<float>();
        
        // Verify against reference implementation
        std::cout << data[0] << " | " << gelu_reference(-2.0f) << std::endl;
        assert(is_close(data[0], gelu_reference(-2.0f)));
        assert(is_close(data[1], gelu_reference(-1.0f)));
        assert(is_close(data[2], gelu_reference(0.0f)));  // GELU(0) ≈ 0
        assert(is_close(data[3], gelu_reference(1.0f)));
        assert(is_close(data[4], gelu_reference(2.0f)));
        
        std::cout << "  GELU(-2) = " << data[0] << " (expected ≈ " << gelu_reference(-2.0f) << ")" << std::endl;
        std::cout << "  GELU(-1) = " << data[1] << " (expected ≈ " << gelu_reference(-1.0f) << ")" << std::endl;
        std::cout << "  GELU(0)  = " << data[2] << " (expected ≈ " << gelu_reference(0.0f) << ")" << std::endl;
        std::cout << "  GELU(1)  = " << data[3] << " (expected ≈ " << gelu_reference(1.0f) << ")" << std::endl;
        std::cout << "  GELU(2)  = " << data[4] << " (expected ≈ " << gelu_reference(2.0f) << ")" << std::endl;
    }

    // Test 2: Positive values (should be close to identity for large x)
    {
        std::cout << "\nTest 2: Large positive values" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}}, options);
        t.set_data({3.0f, 5.0f, 10.0f});

        Tensor out = mlp::GeLU(t);
        out.to_cpu().display();
        
        const float* data = out.to_cpu().data<float>();
        
        // For large positive x, GELU(x) ≈ x
        assert(is_close(data[0], 3.0f, 0.01f));
        assert(is_close(data[1], 5.0f, 0.01f));
        assert(is_close(data[2], 10.0f, 0.01f));
    }

    // Test 3: Negative values (should approach 0 for large negative x)
    {
        std::cout << "\nTest 3: Large negative values" << std::endl;
        Tensor t = Tensor::zeros(Shape{{3}}, options);
        t.set_data({-3.0f, -5.0f, -10.0f});

        Tensor out = mlp::GeLU(t);
        out.to_cpu().display();
        
        const float* data = out.to_cpu().data<float>();
        
        // For large negative x, GELU(x) ≈ 0
        assert(is_close(data[0], 0.0f, 0.01f));
        assert(is_close(data[1], 0.0f, 0.001f));
        assert(is_close(data[2], 0.0f, 0.0001f));
    }

    // Test 4: 2D Tensor Shape
    {
        std::cout << "\nTest 4: 2D Tensor Shape" << std::endl;
        Tensor t = Tensor::zeros(Shape{{2, 2}}, options);
        t.set_data({-1.0f, 0.0f, 0.5f, 1.0f});

        Tensor out = mlp::GeLU(t);
        
        assert(out.shape() == t.shape());
        const float* data = out.to_cpu().data<float>();
        
        assert(is_close(data[0], gelu_reference(-1.0f)));
        assert(is_close(data[1], gelu_reference(0.0f)));
        assert(is_close(data[2], gelu_reference(0.5f)));
        assert(is_close(data[3], gelu_reference(1.0f)));
    }

    // Test 5: Smoothness around zero
    {
        std::cout << "\nTest 5: Smoothness around zero" << std::endl;
        Tensor t = Tensor::zeros(Shape{{5}}, options);
        t.set_data({-0.5f, -0.25f, 0.0f, 0.25f, 0.5f});

        Tensor out = mlp::GeLU(t);
        out.to_cpu().display();
        
        const float* data = out.to_cpu().data<float>();
        
        // Verify smooth transition through zero
        for(int i = 0; i < 5; ++i) {
            float input_val = -0.5f + i * 0.25f;
            assert(is_close(data[i], gelu_reference(input_val)));
        }
        
        // GELU should be smooth (no sharp transitions like ReLU)
        // Values should gradually increase
        assert(data[0] < data[1]);
        assert(data[1] < data[2]);
        assert(data[2] < data[3]);
        assert(data[3] < data[4]);
    }

    // Test 6: Comparison with ReLU-like behavior
    {
        std::cout << "\nTest 6: Comparison with ReLU-like behavior" << std::endl;
        Tensor t = Tensor::zeros(Shape{{4}}, options);
        t.set_data({-2.0f, -0.5f, 0.5f, 2.0f});

        Tensor gelu_out = mlp::GeLU(t);
        Tensor relu_out = mlp::ReLU(t);
        
        const float* gelu_data = gelu_out.to_cpu().data<float>();
        const float* relu_data = relu_out.to_cpu().data<float>();
        
        // For negative values, GELU should be small but non-zero (unlike ReLU)
        assert(gelu_data[0] < 0.0f);
        assert(gelu_data[1] < 0.0f);  // GELU can be slightly negative
        
        // For positive values, GELU should be close to input (like ReLU)
        assert(is_close(gelu_data[2], 0.5f, 0.2f));
        assert(is_close(gelu_data[3], 2.0f, 0.05f));
        
        std::cout << "  GELU vs ReLU at x=-2:  " << gelu_data[0] << " vs " << relu_data[0] << std::endl;
        std::cout << "  GELU vs ReLU at x=-0.5: " << gelu_data[1] << " vs " << relu_data[1] << std::endl;
        std::cout << "  GELU vs ReLU at x=0.5:  " << gelu_data[2] << " vs " << relu_data[2] << std::endl;
        std::cout << "  GELU vs ReLU at x=2:    " << gelu_data[3] << " vs " << relu_data[3] << std::endl;
    }

    std::cout << "\n✅ GELU Verification Passed!" << std::endl;
    return 0;
}
