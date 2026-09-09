"""
SKv3_1: Kv3.1-type potassium conductance 

Extracted from BBP NEURON mechanism: SKv3_1.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
"""

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (S/cm^2)
gSKv3_1bar = 0.00001

def derivatives(m, V):
    """
    Compute steady-state value and time constant,
    then compute time derivative of the gating variable.

    Inputs:
        - m (float): activation gate variable
        - V (float): membrane potential (mV)

    Returns: 
        - dm_dt (float): time derivative of m
    """

    # steady state activation
    mInf = 1.0 / (1.0 + e**((V - 18.7) / -9.7))

    # activation time constant (ms)
    mTau = 0.2 * 20 / (1+e**((V + 46.560) / -44.140))

    dm_dt = (mInf - m) / mTau

    return dm_dt

def current(m, V, E_K, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - m (float): activation gate variable
        - V (float): membrane potential (mV)
        - E_K (float): calcium reversal potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_K (float): current (mA/cm^2)
    """
    # steady state activation
    mInf = 1.0 / (1.0 + e**((V - 18.7) / -9.7))

    # activation time constant (ms)
    mTau = 0.2 * 20 / (1+e**((V + 46.560) / -44.140))
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    
    gSKv3_1 = gSKv3_1bar * m
    I_K = gSKv3_1 * (V - E_K)
    return I_K