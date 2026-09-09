"""Im: muscarinic (M-type) potassium current.

Transcribed from ``Im.mod`` (Reference: Adams et al. 1982).
"""

import numpy as np

from .base import E_K, QT, GateSpec, GatingChannel


class Im(GatingChannel):
    name = "Im"
    ion = "k"
    gates = (GateSpec("m", 1),)
    default_gbar = 0.00001
    default_erev = E_K
    reference = "Adams et al. 1982"

    def _rates(self, v, cai):
        mAlpha = 3.3e-3 * np.exp(2.5 * 0.04 * (v - -35))
        mBeta = 3.3e-3 * np.exp(-2.5 * 0.04 * (v - -35))
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = (1 / (mAlpha + mBeta)) / QT
        return [(mInf, mTau)]
