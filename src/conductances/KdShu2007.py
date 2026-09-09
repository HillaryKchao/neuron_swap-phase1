"""
KdShu2007: KdShu2007 potassium Conductance

Extracted from BBP NEURON mechanism: KdShu2007.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
    h: inactivation gate
"""

VHALF_M = -43.0  # mV
K_M = 8.0

VHALF_H = -67.0  # mV
K_H = 7.3

M_TAU = 0.6  # ms
H_TAU = 1500.0  # ms

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (mho/cm^2)
gKBar = 0.1

# Potassium reversal potential (mV) 
e_K = -100.0

def derivatives(state, V):
    """
    Compute steady-state values and time constants,
    then compute time derivatives of the gating variables.

    Inputs:
        - state (tuple of floats): (m, h) gating variables
        - V (float): membrane potential (mV)

    Returns: 
        - dm_dt (float): time derivative of m
        - dh_dt (float): float time derivative of h
    """

    # note: temperature scaling is defined in original .mod file
    # but it does not affect any variables in this method, so it is omitted here

    # steady state activation & inactivation
    mInf = 1.0 - 1.0 / (1.0 + e**((V - VHALF_M) / K_M))
    hInf = 1.0 / (1.0 + e**((V - VHALF_H) / K_H))

    m, h = state

    dm_dt = (mInf - m) / M_TAU
    dh_dt = (hInf - h) / H_TAU

    return dm_dt, dh_dt

def current(state, V, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - state (tuple of floats): (m, h) gating variables
        - V (float): membrane potential (mV)
        - E_K (float): potassium reversal potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_K (float): current (mA/cm^2)
    """
    m, h = state
    
    # steady state activation & inactivation
    mInf = 1.0 - 1.0 / (1.0 + e**((V - VHALF_M) / K_M))
    hInf = 1.0 / (1.0 + e**((V - VHALF_H) / K_H))

    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/M_TAU)
    h = hInf + (h - hInf) * e**(-dt/H_TAU)

    I_K = gKBar * m * h * (V - e_K)

    return I_K