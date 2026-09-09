#include "SKv3_1.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gSKv3_1bar = 0.00001;
}

scalar_t SKv3_1_derivatives(scalar_t m, scalar_t V) {
    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, ((V - 18.7) / -9.7)));
    const scalar_t mTau = 0.2 * 20.0 / (1.0 + std::pow(e, ((V + 46.560) / -44.140)));

    return (mInf - m) / mTau;
}

scalar_t SKv3_1_current(scalar_t m, scalar_t V, scalar_t E_K, scalar_t dt) {
    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, ((V - 18.7) / -9.7)));
    const scalar_t mTau = 0.2 * 20.0 / (1.0 + std::pow(e, ((V + 46.560) / -44.140)));

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));

    const scalar_t gSKv3_1 = gSKv3_1bar * m;
    const scalar_t I_K = gSKv3_1 * (V - E_K);
    return I_K;
}