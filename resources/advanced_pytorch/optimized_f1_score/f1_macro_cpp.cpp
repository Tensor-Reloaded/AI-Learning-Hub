#include <torch/extension.h>

double f1_score(torch::Tensor x, torch::Tensor y) {
    auto x_sum = x.sum().item<double>();
    auto y_sum = y.sum().item<double>();

    if (x_sum == 0.0 || y_sum == 0.0) {
        if (x_sum == 0.0 && y_sum == 0.0)
            return 1.0;
        return 0.0;
    }

    auto intersection = (x & y).sum().item<double>();
    return 2.0 * intersection / (x_sum + y_sum);
}

double f1_macro(torch::Tensor x, torch::Tensor y, int classes) {
    TORCH_CHECK(x.sizes() == y.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(x.device() == y.device(), "Input tensors must be on the same device");

    double result = 0.0;
    for (int c = 0; c < classes; ++c) {
        auto x_mask = (x == c);
        auto y_mask = (y == c);
        result += f1_score(x_mask, y_mask);
    }
    return result / classes;
}

// Bind functions to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f1_score", &f1_score, "F1 score for one class (bool tensors)");
    m.def("f1_macro", &f1_macro, "Macro F1 score (int tensors)");
}
