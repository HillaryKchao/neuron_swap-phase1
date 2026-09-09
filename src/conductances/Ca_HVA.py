"""
Ca_HVA: High-voltage-activated calcium conductance

Extracted from BBP NEURON mechanism: Ca_HVA.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
    h: inactivation gate
"""

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (S/cm^2)
gCa_HVAbar = 0.00001


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
    if V == -27:
        V += 0.0001
    
    # steady state activation and activation time constant (ms)
    mAlpha =  -(0.055 * (V + 27)) / (e**(-(V + 27) / 3.8) - 1)        
    mBeta  =  (0.94* e**(-(V + 75) / 17))
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = 1 / (mAlpha + mBeta)
    
    # steady state inactivation and inactivation time constant (ms)
    hAlpha =  (0.000457 * e**(-(V + 13) / 50))
    hBeta  =  (0.0065 / (e**(-(V + 15) / 28) + 1))
    hInf = hAlpha / (hAlpha + hBeta)
    hTau = 1 / (hAlpha + hBeta)
    
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
    m, h = state
    
    # voltage shift
    if V == -27:
        V += 0.0001
    
    # steady state activation and activation time constant (ms)
    mAlpha =  -(0.055 * (V + 27)) / (e**(-(V + 27) / 3.8) - 1)        
    mBeta  =  (0.94* e**(-(V + 75) / 17))
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = 1 / (mAlpha + mBeta)
    
    # steady state inactivation and inactivation time constant (ms)
    hAlpha =  (0.000457 * e**(-(V + 13) / 50))
    hBeta  =  (0.0065 / (e**(-(V + 15) / 28) + 1))
    hInf = hAlpha / (hAlpha + hBeta)
    hTau = 1 / (hAlpha + hBeta)
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    h = hInf + (h - hInf) * e**(-dt/hTau)
    
    gCa_HVAst = gCa_HVAbar * m * m * h
    I_Ca = gCa_HVAst * (V - E_Ca)
    return I_Ca