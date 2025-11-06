// File: ops/helpers/SleefWrapper.h
#pragma once
#include <cmath>
#include "dtype/Types.h"

#ifdef WITH_SLEEF
#include "sleef.h"
#endif

namespace OwnTensor {

// ============================================================================
// SLEEF Vectorized Functions with Fallbacks
// ============================================================================

// Exponential functions
template<typename T>
struct SleefExp {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_expf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_exp_u10(x);
        }
#else
        return std::exp(x);
#endif
    }
};

template<typename T>
struct SleefExp2 {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_exp2f_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_exp2_u10(x);
        }
#else
        return std::exp2(x);
#endif
    }
};

// Logarithmic functions
template<typename T>
struct SleefLog {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_logf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_log_u10(x);
        }
#else
        return std::log(x);
#endif
    }
};

template<typename T>
struct SleefLog2 {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_log2f_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_log2_u10(x);
        }
#else
        return std::log2(x);
#endif
    }
};

template<typename T>
struct SleefLog10 {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_log10f_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_log10_u10(x);
        }
#else
        return std::log10(x);
#endif
    }
};

// Trigonometric functions
template<typename T>
struct SleefSin {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_sinf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_sin_u10(x);
        }
#else
        return std::sin(x);
#endif
    }
};

template<typename T>
struct SleefCos {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_cosf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_cos_u10(x);
        }
#else
        return std::cos(x);
#endif
    }
};

template<typename T>
struct SleefTan {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_tanf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_tan_u10(x);
        }
#else
        return std::tan(x);
#endif
    }
};

// Inverse trigonometric functions
template<typename T>
struct SleefAsin {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_asinf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_asin_u10(x);
        }
#else
        return std::asin(x);
#endif
    }
};

template<typename T>
struct SleefAcos {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_acosf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_acos_u10(x);
        }
#else
        return std::acos(x);
#endif
    }
};

template<typename T>
struct SleefAtan {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_atanf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_atan_u10(x);
        }
#else
        return std::atan(x);
#endif
    }
};

// Hyperbolic functions
template<typename T>
struct SleefSinh {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_sinhf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_sinh_u10(x);
        }
#else
        return std::sinh(x);
#endif
    }
};

template<typename T>
struct SleefCosh {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_coshf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_cosh_u10(x);
        }
#else
        return std::cosh(x);
#endif
    }
};

template<typename T>
struct SleefTanh {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_tanhf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_tanh_u10(x);
        }
#else
        return std::tanh(x);
#endif
    }
};

// Inverse hyperbolic functions
template<typename T>
struct SleefAsinh {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_asinhf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_asinh_u10(x);
        }
#else
        return std::asinh(x);
#endif
    }
};

template<typename T>
struct SleefAcosh {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_acoshf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_acosh_u10(x);
        }
#else
        return std::acosh(x);
#endif
    }
};

template<typename T>
struct SleefAtanh {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_atanhf_u10(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_atanh_u10(x);
        }
#else
        return std::atanh(x);
#endif
    }
};

// Square root
template<typename T>
struct SleefSqrt {
    static inline T func(T x) {
#ifdef WITH_SLEEF
        if constexpr (std::is_same_v<T, float>) {
            return Sleef_sqrtf_u05(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return Sleef_sqrt_u05(x);
        }
#else
        return std::sqrt(x);
#endif
    }
};

} // namespace OwnTensor