#include <iostream>
#include "../include/UnaryDispatcher.hpp"

// Implementation of KernelKey operators
bool KernelKey::operator==(const KernelKey& other) const {
    return op == other.op && dtype == other.dtype && device == other.device && mode == other.mode;
}

// Implementation of KernelKeyHash operator
size_t KernelKeyHash::operator()(const KernelKey& key) const noexcept {
    size_t a = static_cast<size_t>(key.op);
    size_t b = static_cast<size_t>(key.dtype);
    size_t c = static_cast<size_t>(key.device);
    size_t d = static_cast<size_t>(key.mode);
    return a ^ (b << 1) ^ (c << 2) ^ (d << 3); 
    // return ((a * 1315423911u) ^ (b + 0x9e3779b97f4a7c15ULL + (a<<6) + (a>>2))) ^ (c << 1);
}

// Implementation of KernelRegistry members
KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry reg;
    return reg;
}

void KernelRegistry::register_kernel(const KernelKey& key, UnaryKernelFn fn) {
    registry_[key] = fn;
    std::cout << "Registered Kernel: "
              << unaryop_to_string(key.op) << "_"
              << dtype_to_string(key.dtype) << "_"
              << device_to_string(key.device) << "_"
              << mode_to_string(key.mode)
              << "\n";
}

UnaryKernelFn KernelRegistry::get_kernel(const KernelKey& key) const {
    auto it = registry_.find(key);
    if (it != registry_.end()) return it->second;
    return nullptr; // always return safely
}