"""
StochKv: Stochastic potassium conductance

Extracted from BBP NEURON mechanism: StochKV.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
"""
import numpy as np

THA = -40.0
QA = 9.0
RA = 0.02
RB = 0.002
Q10 = 2.3
TEMP = 23.0

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (S/cm^2)
gK_bar = 0.75

gamma = 30 # pS

def derivatives(n, V, celsius):
    """
    Compute steady-state value and time constant,
    then compute time derivative of the gating variables.

    Inputs:
        - n (float): gating variable
        - V (float): membrane potential (mV)
        - celsius (float): simulation temperature in degC
        
    Returns: 
        - dn_dt (float): time derivative of n
    """
    # temperature adjustment
    tadj = Q10 ** ((celsius - TEMP) / 10.0)

    # forward and backward rates
    if abs(V - THA) > 1e-6:
        a = RA * (V - THA) / (1.0 - e**((THA - V) / QA))
        b = - RB * (V - THA) / (1.0 - e**((THA - V) / QA))
    else:
        a = RA * QA
        b = RB * QA
    a *= tadj
    b *= tadj

    # dynamics
    nInf = a / (a + b)
    nTau = 1.0 / (a + b)
    dn_dt = (nInf - n) / nTau

    return dn_dt

def current(n, V, celsius, E_K, area, N0, N1, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - n (float): gating variable
        - V (float): membrane potential (mV)
        - celsius (float): simulation temperature in degC
        - E_K (float): calcium reversal potential (mV)
        - area (float): membrane patch area (µm²)
        - N0 (int): number of channels in closed (non-conducting) state
        - N1 (int): number of channels in open (conducting) state
        - time step (float): time step of derivative measurement

    Returns: 
        - I_K (float): current (mA/cm^2)
    """
    # temperature adjustment
    tadj = Q10 ** ((celsius - TEMP) / 10.0)

    # forward and backward rates
    if abs(V - THA) > 1e-6:
        a = RA * (V - THA) / (1.0 - e**((THA - V) / QA))
        b = - RB * (V - THA) / (1.0 - e**((THA - V) / QA))
    else:
        a = RA * QA
        b = RB * QA
    a *= tadj
    b *= tadj
    
    # declaring some constants
    eta = gK_bar / gamma
    scale_dens = gamma / area
    N = np.floor(eta * area + 0.5)
    N1 = np.floor(n * N + 0.5)
    N0 = N - N1

    # transition probabilities
    P_a = max(0.0, min(1.0, a * dt))
    P_b = max(0.0, min(1.0, b * dt))

    # binomial transitions
    n0_n1 = np.random.binomial(N0, P_a)
    n1_n0 = np.random.binomial(N1, P_b)

    # update transitions
    N0_new = max(0, N0 - n0_n1 + n1_n0)
    N1_new = (N0 + N1) - N0_new

    gk = N1_new * scale_dens * tadj

    I_K = 1e-4 * gk * (V - E_K)
    return I_K