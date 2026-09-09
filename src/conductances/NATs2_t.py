"""
NATs2_t: Transient sodium conductance

Extracted from BBP NEURON mechanism: NATs2_t.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
    h: inactivation gate
"""

# temperature correction
QT = 2.3**((34 - 21) / 10)

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (S/cm^2)
gNaTs2_tbar = 0.00001

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

    # voltage shift
    if V == -32 or V == -60:
        V += 0.0001

    # activation
    mAlpha = (0.182 * (V + 32)) / (1 - e**(-(V + 32) / 6))
    mBeta = -(0.124 * (V + 32)) / (1 - e**((V + 32) / 6))
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = (1 / (mAlpha + mBeta)) / QT

    # inactivation
    hAlpha = -(0.015 * (V + 60)) / (1 - e**((V + 60) / 6))
    hBeta = (0.015 * (V + 60)) / (1 - e**(-(V + 60) / 6))
    hInf = hAlpha / (hAlpha + hBeta)
    hTau = (1 / (hAlpha + hBeta)) / QT

    m, h = state

    dm_dt = (mInf - m) / mTau
    dh_dt = (hInf - h) / hTau

    return dm_dt, dh_dt

def current(state, V, E_Na, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - state (tuple of floats): (m, h) gating variables
        - V (float): membrane potential (mV)
        - E_Na (float): sodium reversal potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_Na (float): current (mA/cm^2)
    """
    # voltage shift
    if V == -32 or V == -60:
        V += 0.0001

    # activation
    mAlpha = (0.182 * (V + 32)) / (1 - e**(-(V + 32) / 6))
    mBeta = -(0.124 * (V + 32)) / (1 - e**((V + 32) / 6))
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = (1 / (mAlpha + mBeta)) / QT

    # inactivation
    hAlpha = -(0.015 * (V + 60)) / (1 - e**((V + 60) / 6))
    hBeta = (0.015 * (V + 60)) / (1 - e**(-(V + 60) / 6))
    hInf = hAlpha / (hAlpha + hBeta)
    hTau = (1 / (hAlpha + hBeta)) / QT

    m, h = state
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    h = hInf + (h - hInf) * e**(-dt/hTau)
    
    gNaTs2_t = gNaTs2_tbar * m * m * m * h
    I_Na = gNaTs2_t * (V - E_Na)
    return I_Na