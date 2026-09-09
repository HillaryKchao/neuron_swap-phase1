#include "K_Tst.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t gK_Tstbar = 0.00001;
}

std::pair<scalar_t, scalar_t> K_Tst_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    const double v = V + 10.0;

    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, (-v / 19.0)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((v + 66.0) / 10.0)));

    const scalar_t mTau = (0.34 + 0.92 * std::pow(e, (-std::pow((v + 71.0) / 59.0, 2.0)))) / QT;
    const scalar_t hTau = (8.0 + 49.0 * std::pow(e, (-std::pow((v + 73.0) / 23.0, 2.0)))) / QT;

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t K_Tst_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_K, scalar_t dt) {
    const double v = V + 10.0;

    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, (-v / 19.0)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((v + 66.0) / 10.0)));

    const scalar_t mTau = (0.34 + 0.92 * std::pow(e, (-std::pow((v + 71.0) / 59.0, 2.0)))) / QT;
    const scalar_t hTau = (8.0 + 49.0 * std::pow(e, (-std::pow((v + 73.0) / 23.0, 2.0)))) / QT;

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    h = hInf + (h - hInf) * std::pow(e, (-dt / hTau));

    const scalar_t gK_Tst = gK_Tstbar * m * m * m * m * h;
    const scalar_t I_K = gK_Tst * (V - E_K);
    return I_K;
}