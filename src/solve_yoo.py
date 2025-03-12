import numpy as np
from k_eff import *
from tqdm import tqdm

from utils import *

# Now, we implement the explicit finite difference solver.
def explicit_finite_difference_solver(params):
    """
    Solves the 1D heat conduction equation using an explicit finite difference method.
    
    Governing equation:
      ρ c_p ∂T/∂t = ∂/∂x ( k_eff(x,T) ∂T/∂x )
      
    Boundary conditions (adapted from Table 2):
      - At x = 0 (evaporator wall): prescribed flux (Neumann condition)
          dT/dx |_{x=0} = -q_e'' / k_w
      - In the adiabatic section (L_e < x < L_e+L_a): zero flux, dT/dx = 0.
      - At x = L_t (condenser): radiative heat loss,
          -k_w dT/dx = σ ε (T^4 - T_∞^4)
          (A linearized version is used: T^4 - T_∞^4 ≈ 4*T_∞^3*(T - T_∞))
          
    The sonic limit (Eq. (33)) is applied in the vapor region:
      k_eff_v(T) <= Q_sonic(T) / (A_c * |dT/dx|)
    
    params: a dictionary containing all the necessary parameters.
    """
    
    # Unpack simulation parameters:
    L_t   = params["L_t"]       # Total length [m]
    L_e   = params["L_e"]       # Evaporator length [m]
    L_a   = params["L_a"]       # Adiabatic length [m]
    L_c   = params["L_c"]       # Condenser length [m]
    dx    = params["dx"]        # Spatial step [m]
    dt    = params["dt"]        # Time step [s]
    t_end = params["t_end"]     # End time [s]
    
    # Material and fluid properties
    rho   = params["rho"]       # Density [kg/m^3]
    c_p   = params["c_p"]       # Specific heat [J/(kg-K)]
    
    # For wall boundary conditions:
    k_w   = params["k_w"]       # Thermal conductivity of the wall [W/(m-K)]
    q_e   = params["q_e"]       # Heat flux at evaporator [W/m^2]
    sigma = params["sigma"]     # Stefan-Boltzmann constant [W/(m^2-K^4)]
    eps   = params["epsilon"]   # Emissivity (dimensionless)
    T_inf = params["T_inf"]     # Ambient temperature [K]
    
    # Vapor effective conductivity parameters (for compute_k_eff_with_M_g):
    P       = params["P"]       # Saturated vapor pressure [Pa]
    R_v     = params["R_v"]     # Vapor core radius [m]
    mu_v    = params["mu_v"]    # Dynamic viscosity [Pa-s]
    m_g     = params["m_g"]     # Molecular mass [kg]
    k_B     = params["k_B"]     # Boltzmann constant [J/K]
    R_g     = params["R_g"]     # Specific gas constant [J/(kg-K)]
    N_A     = params["N_A"]     # Avogadro's number [1/mol]
    h_lv    = params["h_lv"]    # Latent heat of vaporization [J/kg]
    h_l     = params["h_l"]     # Latent heat coefficient (evap/con)
    h_v     = params["h_v"]     # Vapor enthalpy (adiabatic)
    M_g_val = params["M_g"]     # Molar mass of vapor [kg/mol]
    
    # Sonic limit parameters (for vapor region)
    A_c    = params["A_c"]      # Cross-sectional area of vapor core [m^2]
    rho_v0 = params["rho_v0"]   # Vapor density at sonic condition [kg/m^3]
    h_cl   = params["h_cl"]     # Characteristic latent heat (or similar) [J/kg]
    gamma  = params["gamma"]    # Specific heat ratio (adiabatic index)
    T_v0   = params["T_v0"]     # Reference vapor temperature for sonic limit [K]
    
    # Calculate Q_sonic (from Levy's equation, Eq. (31))
    Q_sonic = rho_v0 * A_c * h_cl * np.sqrt((gamma * R_g * T_v0) / (2.0 * (gamma + 1)))
    
    # Create spatial grid
    N = int(L_t/dx) + 1
    x = np.linspace(0, L_t, N)
    
    # Initialize temperature field. For simplicity, assume a single initial condition for the entire domain.
    T = np.full(N, params["T_init"])
    
    # For storing time history if desired (here we just update T)
    t = 0.0
    n_steps = int(t_end/dt)
    
    # Create an array to store temperature history (size: time steps × spatial points)
    T_history = np.zeros((n_steps + 1, N))

    # Store the initial temperature profile
    T_history[0, :] = T

    # Main time stepping loop
    for step in tqdm(range(n_steps), desc="Solving heat equation"):
        T_new = T.copy()
        
        # Update interior nodes (i=1 to N-1)
        for i in range(1, N-1):
            # Determine region for node i based on its x-position
            region = get_region(x[i], L_e, L_a, L_t)
            
            # Compute effective thermal conductivity at node i using our conduction model
            # (Assume that the vapor region uses the conduction-based k_eff)
            k_eff_i = compute_k_eff(T[i], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                             h_lv, h_l, h_v, region)
            
            # For interface values, a simple arithmetic average is used.
            # For node i+1:
            region_right = get_region(x[i+1], L_e, L_a, L_t)
            k_eff_ip = compute_k_eff(T[i+1], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                              h_lv, h_l, h_v, region_right)
            k_eff_right = 0.5*(k_eff_i + k_eff_ip)
            
            # For node i-1:
            region_left = get_region(x[i-1], L_e, L_a, L_t)
            k_eff_im = compute_k_eff(T[i-1], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                              h_lv, h_l, h_v, region_left)
            k_eff_left = 0.5*(k_eff_i + k_eff_im)
            
            # Compute temperature gradients at interfaces:
            dT_dx_right = (T[i+1] - T[i]) / dx
            dT_dx_left  = (T[i] - T[i-1]) / dx
            
            # In the vapor region, enforce the sonic limit on k_eff.
            # Here, we assume that nodes in the vapor core are those using the conduction-based effective conductivity.
            # If the region is vapor (i.e. 'evap_con' or 'adiabatic' as defined), apply sonic limit.
            k_eff_right = enforce_sonic_limit(k_eff_right, dT_dx_right, Q_sonic, A_c)
            k_eff_left  = enforce_sonic_limit(k_eff_left, dT_dx_left, Q_sonic, A_c)
            
            # Explicit finite difference update:
            T_new[i] = T[i] + (dt / (rho * c_p * (dx**2))) * \
                       ( k_eff_right * (T[i+1] - T[i]) - k_eff_left * (T[i] - T[i-1]) )
        
        # Boundary condition at x = 0 (evaporator end) using ghost node approach:
        # 1. Create a ghost node T_ghost at x = -dx
        # 2. Apply central difference: (T[1] - T_ghost)/(2*dx) = -q_e/k_w
        # 3. Solve for T_ghost: T_ghost = T[1] + 2*dx*q_e/k_w
        T_ghost = T[1] + 2*dx*q_e/k_w
        
        # Determine region for node 0
        region = get_region(x[0], L_e, L_a, L_t)
        
        # Compute k_eff for the boundary node
        k_eff_0 = compute_k_eff(T[0], P, R_v, mu_v, m_g, k_B, R_g, N_A, 
                                h_lv, h_l, h_v, region)
        
        # Compute k_eff for the next node
        region_right = get_region(x[1], L_e, L_a, L_t)
        k_eff_1 = compute_k_eff(T[1], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                h_lv, h_l, h_v, region_right)
        
        # Interface conductivity
        k_eff_right = 0.5*(k_eff_0 + k_eff_1)
        
        # Temperature gradient at the right interface
        dT_dx_right = (T[1] - T[0]) / dx
        
        # Apply sonic limit
        k_eff_right = enforce_sonic_limit(k_eff_right, dT_dx_right, Q_sonic, A_c)
        
        # Update the boundary node using the standard explicit scheme with the ghost node
        T_new[0] = T[0] + (dt / (rho * c_p * (dx**2))) * \
                   (k_eff_right * (T[1] - T[0]) - k_eff_0 * (T[0] - T_ghost))
        
        # Boundary condition at x = L_t (condenser end):
        # Radiative flux: -k_w*(dT/dx)|_{L_t} = σ ε (T^4 - T_inf^4)
        # Linearize: T^4 - T_inf^4 ≈ 4*T_inf^3*(T - T_inf)
        
        # 1. Define the linearized heat transfer coefficient
        h_r = 4 * sigma * eps * (T_inf**3)
        
        # 2. Define ghost node location at x = L_t + dx and determine T[N]
        # For radiative BC: -k_w * (T[N] - T[N-2])/(2*dx) = h_r * (T[N-1] - T_inf)
        # Solve for T[N]: T[N] = T[N-2] - (2*dx*h_r/k_w)*(T[N-1] - T_inf)
        T_ghost_end = T[-2] - (2*dx*h_r/k_w)*(T[-1] - T_inf)
        
        # Determine region for the last node
        region_end = get_region(x[-1], L_e, L_a, L_t)
        
        # Compute k_eff for the boundary node
        k_eff_end = compute_k_eff(T[-1], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                 h_lv, h_l, h_v, region_end)
        
        # Compute k_eff for the previous node
        region_left_end = get_region(x[-2], L_e, L_a, L_t)
        k_eff_prev = compute_k_eff(T[-2], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                  h_lv, h_l, h_v, region_left_end)
        
        # Interface conductivity
        k_eff_left_end = 0.5*(k_eff_end + k_eff_prev)
        
        # Temperature gradient at the left interface
        dT_dx_left_end = (T[-1] - T[-2]) / dx
        
        # Apply sonic limit
        k_eff_left_end = enforce_sonic_limit(k_eff_left_end, dT_dx_left_end, Q_sonic, A_c)
        
        # Update the boundary node using the standard explicit scheme with the ghost node
        T_new[-1] = T[-1] + (dt / (rho * c_p * (dx**2))) * \
                    (k_eff_end * (T_ghost_end - T[-1]) - k_eff_left_end * (T[-1] - T[-2]))
        
        # Store the new temperature profile
        T_history[step + 1, :] = T_new

        # Update temperature field and time
        T = T_new.copy()
        t += dt
    
    return x, T_history