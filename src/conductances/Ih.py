"""
Im: HCN Conductance

Extracted from BBP NEURON mechanism: Ih.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
"""

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (S/cm^2)
gIh_bar = 0.00001

# HCN reversal potential (mV) 
e_HCN = -45.0

def derivatives(m, V):
    """
    Compute steady-state value and time constant,
    then compute time derivative of the gating variable.

    Inputs:
        - m (float): gating variable
        - V (float): membrane potential (mV)

    Returns: 
        - dm_dt (float): time derivative of m
    """
    
    # voltage shift
    if V == -154.9:
        V += 0.0001

    mAlpha = 0.001 * 6.43 * (V + 154.9) / (e**((V + 154.9) / 11.9) - 1)
    mBeta = 0.001 * 193 * e**(V / 33.1)
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = (1 / (mAlpha + mBeta))

    dm_dt = (mInf - m) / mTau

    return dm_dt

def current(m, V, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - m (float): gating variable
        - V (float): membrane potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_HCN (float): current (mA/cm^2)
    """
    # voltage shift
    if V == -154.9:
        V += 0.0001

    mAlpha = 0.001 * 6.43 * (V + 154.9) / (e**((V + 154.9) / 11.9) - 1)
    mBeta = 0.001 * 193 * e**(V / 33.1)
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = (1 / (mAlpha + mBeta))
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    
    gIh = gIh_bar * m
    I_HCN = gIh * (V - e_HCN)
    return I_HCN