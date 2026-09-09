"""Ih: hyperpolarisation-activated (HCN) non-specific cation current.

Transcribed from ``Ih.mod`` (Reference: Kole, Hallermann and Stuart 2006).
No Q10 correction is applied in the .mod file.
"""

import numpy as np

from .base import E_HCN, GateSpec, GatingChannel, _guard


class Ih(GatingChannel):
    name = "Ih"
    ion = "hcn"
    gates = (GateSpec("m", 1),)
    default_gbar = 0.00001
    default_erev = E_HCN
    reference = "Kole, Hallermann and Stuart 2006"

    def _rates(self, v, cai):
        v = _guard(v, -154.9)
        mAlpha = 0.001 * 6.43 * (v + 154.9) / (np.exp((v + 154.9) / 11.9) - 1)
        mBeta = 0.001 * 193 * np.exp(v / 33.1)
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = 1 / (mAlpha + mBeta)
        return [(mInf, mTau)]
