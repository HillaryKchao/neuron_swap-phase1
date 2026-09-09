"""
K_Tst: Fast potassium conductance

Extracted from BBP NEURON mechanism: K_Tst.mod
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
gK_Tstbar = 0.00001

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
    mInf = 1.0 / (1.0 + e**(-v / 19.0))
    hInf = 1.0 / (1.0 + e**((v + 66.0) / 10.0))

    # activation & inactivation time constants (ms)
    mTau = (0.34 + 0.92 * e**(-((v + 71) / 59)**2)) / QT
    hTau = (8.0 + 49.0 * e**(-((v + 73) / 23)**2)) / QT

    m, h = state

    dm_dt = (mInf - m) / mTau
    dh_dt = (hInf - h) / hTau

    return dm_dt, dh_dt

def current(state, V, E_K, dt=0.025):
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
        # voltage shift
    v = V + 10.0

    # steady state activation & inactivation
    mInf = 1.0 / (1.0 + e**(-v / 19.0))
    hInf = 1.0 / (1.0 + e**((v + 66.0) / 10.0))

    # activation & inactivation time constants (ms)
    mTau = (0.34 + 0.92 * e**(-((v + 71) / 59)**2)) / QT
    hTau = (8.0 + 49.0 * e**(-((v + 73) / 23)**2)) / QT

    m, h = state
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    h = hInf + (h - hInf) * e**(-dt/hTau)
    
    gK_Tst = gK_Tstbar * m * m * m * m * h
    I_K = gK_Tst * (V - E_K)
    return I_K