"""
Ca_LVAst: Low-voltage-activated calcium conductance

Extracted from BBP NEURON mechanism: Ca_LVAst.mod
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
gCa_LVAstbar = 0.00001

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
    v = V + 10.0

    # steady state activation & inactivation
    mInf = 1.0 / (1.0 + e**((v + 30.0) / -6.0))
    hInf = 1.0 / (1.0 + e**((v + 80.0) / 6.4))

    # activation & inactivation time constants (ms)
    mTau = (5.0 + 20.0 / (1.0 + e**((v + 25.0) / 5.0))) / QT
    hTau = (20.0 + 50.0 / (1.0 + e**((v + 40.0) / 7.0))) / QT

    m, h = state

    dm_dt = (mInf - m) / mTau
    dh_dt = (hInf - h) / hTau

    return dm_dt, dh_dt

def current(state, V, E_Ca, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - state (tuple of floats): (m, h) gating variables
        - V (float): membrane potential (mV)
        - E_Ca (float): calcium reversal potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_Ca (float): current (mA/cm^2)
    """
    # voltage shift
    v = V + 10.0

    # steady state activation & inactivation
    mInf = 1.0 / (1.0 + e**((v + 30.0) / -6.0))
    hInf = 1.0 / (1.0 + e**((v + 80.0) / 6.4))

    # activation & inactivation time constants (ms)
    mTau = (5.0 + 20.0 / (1.0 + e**((v + 25.0) / 5.0))) / QT
    hTau = (20.0 + 50.0 / (1.0 + e**((v + 40.0) / 7.0))) / QT

    m, h = state
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    h = hInf + (h - hInf) * e**(-dt/hTau)
    
    gCa_LVAst = gCa_LVAstbar * m * m * h
    I_Ca = gCa_LVAst * (V - E_Ca)
    return I_Ca