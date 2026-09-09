"""NaTs2_t: fast transient sodium current (somato-dendritic variant).

Transcribed from ``NaTs2_t.mod`` (Reference: Colbert and Pan 2002; NaTa
shifted by 6 mV for both activation and inactivation).
"""

import numpy as np

from .base import E_NA, QT, GateSpec, GatingChannel, _guard


class NaTs2_t(GatingChannel):
    name = "NaTs2_t"
    ion = "na"
    gates = (GateSpec("m", 3), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_NA
    reference = "Colbert and Pan 2002 (NaTa shifted by 6 mV)"

    def _rates(self, v, cai):
        v = _guard(v, -32.0)
        mAlpha = (0.182 * (v - -32)) / (1 - np.exp(-(v - -32) / 6))
        mBeta = (0.124 * (-v - 32)) / (1 - np.exp(-(-v - 32) / 6))
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = (1 / (mAlpha + mBeta)) / QT

        v = _guard(v, -60.0)
        hAlpha = (-0.015 * (v - -60)) / (1 - np.exp((v - -60) / 6))
        hBeta = (-0.015 * (-v - 60)) / (1 - np.exp((-v - 60) / 6))
        hInf = hAlpha / (hAlpha + hBeta)
        hTau = (1 / (hAlpha + hBeta)) / QT
        return [(mInf, mTau), (hInf, hTau)]
