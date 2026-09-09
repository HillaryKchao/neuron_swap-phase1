"""SKv3_1: Kv3.1 (Shaw-related) fast delayed-rectifier potassium current.

Transcribed from ``SKv3_1.mod`` (Reference: Rettig et al., EMBO J 1992).
"""

import numpy as np

from .base import E_K, GateSpec, GatingChannel


class SKv3_1(GatingChannel):
    name = "SKv3_1"
    ion = "k"
    gates = (GateSpec("m", 1),)
    default_gbar = 0.00001
    default_erev = E_K
    reference = "Rettig et al. 1992 (Shaw-related K channel family)"

    def _rates(self, v, cai):
        mInf = 1 / (1 + np.exp((v - 18.700) / -9.700))
        mTau = 0.2 * 20.000 / (1 + np.exp((v - -46.560) / -44.140))
        return [(mInf, mTau)]
