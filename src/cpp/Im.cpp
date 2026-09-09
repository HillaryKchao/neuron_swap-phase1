#include "Im.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t gIm_bar = 0.00001;
}

scalar_t Im_derivatives(scalar_t m, scalar_t V) {
    const scalar_t mAlpha = 3.3e-3 * std::pow(e, (2.5 * 0.04 * (V + 35)));
    const scalar_t mBeta = 3.3e-3 * std::pow(e, (-2.5 * 0.04 * (V + 35)));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = (1.0 / (mAlpha + mBeta)) / QT;

    return (mInf - m) / mTau;
}

scalar_t Im_current(scalar_t m, scalar_t V, scalar_t E_K, scalar_t dt) {
    const scalar_t mAlpha = 3.3e-3 * std::pow(e, (2.5 * 0.04 * (V + 35)));
    const scalar_t mBeta = 3.3e-3 * std::pow(e, (-2.5 * 0.04 * (V + 35)));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = (1.0 / (mAlpha + mBeta)) / QT;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    const scalar_t gIm = gIm_bar * m;

    const scalar_t I_K = gIm * (V - E_K);
    return I_K;
}