"""neuron_swap: Python "Golden Model" of L5PC ion channels plus voltage-clamp
protocols and DCA-based dynamical signatures (Phase 1 of the roadmap).

The channel equations are transcribed from the Blue Brain Project L5PC
(cADpyr232_L5_TTPC1) NEURON mechanisms; see ``neuron_swap.channels``.
"""

from .channels import CHANNELS, get_channel  # noqa: F401

__version__ = "0.1.0"
