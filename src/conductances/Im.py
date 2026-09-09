"""
Im: M-type potasssium Conductance

Extracted from BBP NEURON mechanism: Im.mod
Neuron type: pyramidal neuron

State variables:
    m: activation gate
"""

# temperature correction
QT = 2.3**((34 - 21) / 10)

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (S/cm^2)
gIm_bar = 0.00001

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

    mAlpha = 3.3e-3 * e**(2.5 * 0.04 * (V + 35))
    mBeta = 3.3e-3 * e**(-2.5 * 0.04 * (V + 35))
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = (1 / (mAlpha + mBeta)) / QT

    dm_dt = (mInf - m) / mTau

    return dm_dt

def current(m, V, E_K, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - m (float): gating variable
        - V (float): membrane potential (mV)
        - E_K (float): potassium reversal potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_K (float): current (mA/cm^2)
    """
    mAlpha = 3.3e-3 * e**(2.5 * 0.04 * (V + 35))
    mBeta = 3.3e-3 * e**(-2.5 * 0.04 * (V + 35))
    mInf = mAlpha / (mAlpha + mBeta)
    mTau = (1 / (mAlpha + mBeta)) / QT
    
    # cnexp update
    m = mInf + (m - mInf) * e**(-dt/mTau)
    gIm = gIm_bar * m
    
    I_K = gIm * (V - E_K)
    return I_K