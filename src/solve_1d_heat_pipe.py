import numpy as np
from tqdm import tqdm

def solve_1d_heat_pipe(T, alpha, dx, dt, steps, heat_flux, cooling_coeff):
    """
    T: initial temperature array
    alpha: thermal diffusivity
    dx: spatial step
    dt: time step
    steps: number of time steps
    heat_flux: heat influx at the first quarter (W/m²)
    cooling_coeff: cooling coefficient at the last quarter (W/(m²·K))
    
    Returns: Array of temperature distributions at each time step
    """
    nx = len(T)
    heat_region_end = nx // 4
    middle_region_start = heat_region_end + 1
    middle_region_end = 3 * nx // 8 - 1
    cooling_region_start = middle_region_end + 1  # Fixed cooling start
    
    T_history = np.zeros((steps + 1, nx))
    T_history[0] = T.copy()
    
    for i in tqdm(range(steps), desc="Solving 1D Heat Pipe"):
        Tn = T_history[i].copy()
        T_next = Tn.copy()
        
        # Interior points
        for j in range(1, nx-1):
            if middle_region_start <= j <= middle_region_end:
                # Adiabatic boundary condition (zero heat flux)
                T_next[j] = (Tn[j-1] + Tn[j+1]) / 2
            else:
                # Standard heat equation
                T_next[j] = Tn[j] + alpha * dt/dx**2 * (Tn[j+1] - 2*Tn[j] + Tn[j-1])
        
        # Left boundary: Neumann condition (heat flux)
        T_ghost_left = Tn[1] - (2 * dx * heat_flux / alpha)
        T_next[0] = Tn[0] + alpha * dt/dx**2 * (Tn[1] - 2*Tn[0] + T_ghost_left)
        
        # Apply heat influx (normalized by region size)
        T_next[1:heat_region_end+1] += dt * heat_flux / (dx * heat_region_end)
        
        # Apply cooling in the last quarter
        T_next[cooling_region_start:nx-1] -= dt * cooling_coeff * Tn[cooling_region_start:nx-1]
        
        # Right boundary: Robin condition (cooling)
        T_ghost_right = (Tn[-2] + 2 * dx * cooling_coeff * Tn[-1]) / (1 + 2 * dx * cooling_coeff)
        T_next[-1] = Tn[-1] + alpha * dt/dx**2 * (T_ghost_right - 2*Tn[-1] + Tn[-2])
        
        T_history[i+1] = T_next
    
    return T_history