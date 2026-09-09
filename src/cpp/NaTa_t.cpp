#include "NaTa_t.hpp"

#include <cmath>

namespace {
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gNaTa_tbar = 0.00001;
}

std::pair<scalar_t, scalar_t> NaTa_t_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    if (V == -38) {
        V += 0.0001;
    }

    const scalar_t mAlpha = (0.182 * (V + 38)) / (1 - std::pow(e, (-(V + 38) / 6)));
    const scalar_t mBeta = -(0.124 * (V + 38)) / (1 - std::pow(e, ((V + 38) / 6)));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = (1.0 / (mAlpha + mBeta)) / QT;

    if (V == -66) {
        V += 0.0001;
    }

    const scalar_t hAlpha = -(0.015 * (V + 66)) / (1 - std::pow(e, ((V + 66) / 6)));
    const scalar_t hBeta = (0.015 * (V + 66)) / (1 - std::pow(e, (-(V + 66) / 6)));
    const scalar_t hInf = hAlpha / (hAlpha + hBeta);
    const scalar_t hTau = (1.0 / (hAlpha + hBeta)) / QT;

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t NaTa_t_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Na, scalar_t dt) {
    if (V == -38) {
        V += 0.0001;
    }

    const scalar_t mAlpha = (0.182 * (V + 38)) / (1 - std::pow(e, (-(V + 38) / 6)));
    const scalar_t mBeta = -(0.124 * (V + 38)) / (1 - std::pow(e, ((V + 38) / 6)));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = (1.0 / (mAlpha + mBeta)) / QT;

    if (V == -66) {
        V += 0.0001;
    }

    const scalar_t hAlpha = -(0.015 * (V + 66)) / (1 - std::pow(e, ((V + 66) / 6)));
    const scalar_t hBeta = (0.015 * (V + 66)) / (1 - std::pow(e, (-(V + 66) / 6)));
    const scalar_t hInf = hAlpha / (hAlpha + hBeta);
    const scalar_t hTau = (1.0 / (hAlpha + hBeta)) / QT;

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    h = hInf + (h - hInf) * std::pow(e, (-dt / hTau));

    const scalar_t gNaTa_t = gNaTa_tbar * m * m * m * h;
    const scalar_t I_Na = gNaTa_t * (V - E_Na);
    return I_Na;
}