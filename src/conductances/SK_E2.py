"""
SK_E2: Calcium-activated potassium conductance

Extracted from BBP NEURON mechanism: SK_E2.mod
Neuron type: pyramidal neuron

State variables:
    z: calcium-driven activation gate
"""

Z_TAU = 1.0  # ms
CA_FLOOR = 1e-7  # mM

# euler's constant to 30 digits of precision
e = 2.718281828459045235360287471352

# maximal peak conductance density of channel per unit membrane area (mho/cm^2)
gSK_E2bar = 0.000001

def derivatives(z, ca):
    """
    Compute steady-state value,
    then compute time derivative.

    Inputs:
        - z (float): calcium-driven activation gate
        - ca (float): intracellular calcium concentration (mM)

    Returns: 
        - dz_dt (float): time derivative of z
    """

    if ca < CA_FLOOR:
        ca = ca + CA_FLOOR

    zInf = 1.0 / (1.0 + (0.00043 / ca)**4.8)

    dz_dt = (zInf - z) / Z_TAU
    return dz_dt

def current(z, V, E_K, dt=0.025):
    """
    Compute current based on gating variables and membrane potential.

    Inputs:
        - z (float): calcium-driven activation gate
        - V (float): membrane potential (mV)
        - E_K (float): calcium reversal potential (mV)
        - time step (float): time step of derivative measurement

    Returns: 
        - I_K (float): current (mA/cm^2)
    """
    if ca < CA_FLOOR:
        ca = ca + CA_FLOOR

    zInf = 1.0 / (1.0 + (0.00043 / ca)**4.8)
    
    # cnexp update
    z = zInf + (z - zInf) * e**(-dt/Z_TAU)
    
    gSK_E2 = gSK_E2bar * z
    I_K = gSK_E2 * (E_K)
    return I_K