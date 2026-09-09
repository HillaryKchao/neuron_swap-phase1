"""
Nap_Et2: Persistent sodium conductance

Extracted from BBP NEURON mechanism: Nap_Et2.mod
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
gNap_Et2bar = 0.00001

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

    # steady state activation
    mInf = 1.0 / (1.0 + e**((V + 52.6) / -4.6))

    # voltage shift
    if V == -38:
        V += 0.0001

    # activation time constant
    mAlpha = (0.182 * (V + 38)) / (1 - (e**(-(V + 38) / 6)))
    mBeta = -(0.124 * (V + 38)) / (1 - (e**((V + 38) / 6)))
    mTau = 6 * (1 / (mAlpha + mBeta)) / QT
    
    # more voltage shifts
    if V == -17 or V == -64.4:
        V += 0.001

    # steady state inactivation and inactivation time constants (ms)
    hInf = 1.0 / (1 + e**((V + 48.8) / 10))
    hAlpha = -2.88e-6 * (V + 17) / (1 - e**((V + 17) / 4.63))
    hBeta = 6.94e-6 * (V + 64.4) / (1 - e**(-(V + 64.4) / 2.63))
    hTau = (1 / (hAlpha + hBeta)) / QT

    m, h = state

    dm_dt = (mInf - m) / mTau
    dh_dt = (hInf - h) / hTau

    return dm_dt, dh_dt

def current(state, V, E_Na):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - state (tuple of floats): (m, h) gating variables
        - V (float): membrane potential (mV)
        - E_Na (float): sodium reversal potential (mV)

    Returns: 
        - I_Na (float): current (mA/cm^2)
    """
    m, h = state
    
    gNap_Et2 = gNap_Et2bar * m * m * m * h
    I_Na = gNap_Et2 * (V - E_Na)
    return I_Na