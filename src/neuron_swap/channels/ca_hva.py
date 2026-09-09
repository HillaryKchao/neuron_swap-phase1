"""Ca_HVA: high-voltage-activated calcium current.

Transcribed from ``Ca_HVA.mod`` (Reference: Reuveni, Friedman, Amitai and
Gutnick 1993).  No Q10 correction is applied in the .mod file.
"""

import numpy as np

from .base import E_CA, GateSpec, GatingChannel, _guard


class Ca_HVA(GatingChannel):
    name = "Ca_HVA"
    ion = "ca"
    gates = (GateSpec("m", 2), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_CA
    reference = "Reuveni, Friedman, Amitai and Gutnick 1993"

    def _rates(self, v, cai):
        v = _guard(v, -27.0)
        mAlpha = (0.055 * (-27 - v)) / (np.exp((-27 - v) / 3.8) - 1)
        mBeta = 0.94 * np.exp((-75 - v) / 17)
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = 1 / (mAlpha + mBeta)
        hAlpha = 0.000457 * np.exp((-13 - v) / 50)
        hBeta = 0.0065 / (np.exp((-v - 15) / 28) + 1)
        hInf = hAlpha / (hAlpha + hBeta)
        hTau = 1 / (hAlpha + hBeta)
        return [(mInf, mTau), (hInf, hTau)]
