#include "Ih.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gIh_bar = 0.00001;
const scalar_t e_HCN = -45.0;
}

scalar_t Ih_derivatives(scalar_t m, scalar_t V) {
    if (V == -154.9) {
        V += 0.0001;
    }

    const scalar_t mAlpha = 0.001 * 6.43 * (V + 154.9) / (std::pow(e, ((V + 154.9) / 11.9)) - 1);
    const scalar_t mBeta = 0.001 * 193 * std::pow(e, (V / 33.1));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = 1.0 / (mAlpha + mBeta);

    return (mInf - m) / mTau;
}

scalar_t Ih_current(scalar_t m, scalar_t V, scalar_t dt) {
    if (V == -154.9) {
        V += 0.0001;
    }

    const scalar_t mAlpha = 0.001 * 6.43 * (V + 154.9) / (std::pow(e, ((V + 154.9) / 11.9)) - 1);
    const scalar_t mBeta = 0.001 * 193 * std::pow(e, (V / 33.1));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = 1.0 / (mAlpha + mBeta);

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));

    const scalar_t gIh = gIh_bar * m;
    const scalar_t I_HCN = gIh * (V - e_HCN);
    return I_HCN;
}