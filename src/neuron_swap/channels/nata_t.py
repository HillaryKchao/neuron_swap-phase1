"""NaTa_t: fast transient sodium current (axonal variant).

Transcribed from ``NaTa_t.mod`` (Reference: Colbert and Pan 2002).
"""

import numpy as np

from .base import E_NA, QT, GateSpec, GatingChannel, _guard


class NaTa_t(GatingChannel):
    name = "NaTa_t"
    ion = "na"
    gates = (GateSpec("m", 3), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_NA
    reference = "Colbert and Pan 2002"

    def _rates(self, v, cai):
        v = _guard(v, -38.0)
        mAlpha = (0.182 * (v - -38)) / (1 - np.exp(-(v - -38) / 6))
        mBeta = (0.124 * (-v - 38)) / (1 - np.exp(-(-v - 38) / 6))
        mTau = (1 / (mAlpha + mBeta)) / QT
        mInf = mAlpha / (mAlpha + mBeta)

        v = _guard(v, -66.0)
        hAlpha = (-0.015 * (v - -66)) / (1 - np.exp((v - -66) / 6))
        hBeta = (-0.015 * (-v - 66)) / (1 - np.exp((-v - 66) / 6))
        hTau = (1 / (hAlpha + hBeta)) / QT
        hInf = hAlpha / (hAlpha + hBeta)
        return [(mInf, mTau), (hInf, hTau)]
