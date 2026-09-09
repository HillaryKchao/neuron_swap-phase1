#include "Ca_LVAst.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t gCa_LVAstbar = 0.00001;
}

std::pair<scalar_t, scalar_t> Ca_LVAst_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    const double v = V + 10.0;

    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, ((v + 30.0) / -6.0)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((v + 80.0) / 6.4)));

    const scalar_t mTau = (5.0 + 20.0 / (1.0 + std::pow(e, ((v + 25.0) / 5.0)))) / QT;
    const scalar_t hTau = (20.0 + 50.0 / (1.0 + std::pow(e, ((v + 40.0) / 7.0)))) / QT;

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t Ca_LVAst_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Ca, scalar_t dt) {
    const double v = V + 10.0;

    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, ((v + 30.0) / -6.0)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((v + 80.0) / 6.4)));

    const scalar_t mTau = (5.0 + 20.0 / (1.0 + std::pow(e, ((v + 25.0) / 5.0)))) / QT;
    const scalar_t hTau = (20.0 + 50.0 / (1.0 + std::pow(e, ((v + 40.0) / 7.0)))) / QT;

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    h = hInf + (h - hInf) * std::pow(e, (-dt / hTau));

    const scalar_t gCa_LVAst = gCa_LVAstbar * m * m * h;
    const scalar_t I_Ca = gCa_LVAst * (V - E_Ca);
    return I_Ca;
}