import numpy as np

def solve_yoo(T_init, T_inf, emissivity, heat_input, heat_flux_input, dx, dt, steps):
    """
    Solve the 1D heat equation with heat input and heat flux input.
    
    Parameters
    ----------
    T_init : numpy.ndarray
        Initial temperature profile.
    T_inf : float
        Ambient temperature.
    emissivity : float
        Emissivity of the material.
    heat_input : float
        Heat input to the system.
    heat_flux_input : float
        Heat flux input to the system.
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    steps : int
        Number of time steps to take.
        
    Returns
    -------
    numpy.ndarray
        Temperature profile at each time step.
    """
    # Initialize temperature profile
    T = T_init.copy()
    
    # Pre-calculate coefficients
    alpha = emissivity * heat_flux_input
    beta = heat_input
    
    # Initialize history
    T_history = np.zeros((steps, len(T)))
    
    for i in range(steps):
        # Update temperature profile
        T[1:-1] += alpha * (T[:-2] - 2*T[1:-1] + T[2:]) / dx**2 * dt + beta * dt
        
        # Boundary conditions
        T[0] = T_inf
        T[-1] = T[-2]
        
        # Store temperature profile
        T_history[i] = T
    
    return T_history