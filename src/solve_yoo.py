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
        
        # Compute regions and effective conductivity for all nodes
        regions = np.array([get_region(xi, L_e, L_a, L_t) for xi in x])
        k_eff_all = np.array([compute_k_eff(T[i], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                            h_lv, h_l, h_v, regions[i])
                            for i in range(N)])
        
        # Compute interface conductivities as the arithmetic average
        k_eff_interface = 0.5 * (k_eff_all[1:] + k_eff_all[:-1])
        dT_dx_interface = (T[1:] - T[:-1]) / dx

        # Apply the sonic limit in a vectorized way
        vec_enforce_sonic = np.vectorize(enforce_sonic_limit)
        k_eff_interface = vec_enforce_sonic(k_eff_interface, dT_dx_interface, Q_sonic, A_c)

        # Update interior nodes:
        # For node i, left interface is k_eff_interface[i-1] and right interface is k_eff_interface[i]
        T_new[1:-1] = T[1:-1] + (dt / (rho * c_p * dx**2)) * (
                            k_eff_interface[1:] * (T[2:] - T[1:-1]) -
                            k_eff_interface[:-1] * (T[1:-1] - T[:-2]))
        
        # --- Left boundary (x = 0) ---
        T_ghost = ghost_node_evaporator(T[1], dx, q_e, k_w)
        region0 = get_region(x[0], L_e, L_a, L_t)
        region1 = get_region(x[1], L_e, L_a, L_t)
        k_eff_0 = compute_k_eff(T[0], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                h_lv, h_l, h_v, region0)
        k_eff_1 = compute_k_eff(T[1], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                h_lv, h_l, h_v, region1)
        k_eff_right_bound = 0.5 * (k_eff_0 + k_eff_1)
        dT_dx_bound = (T[1] - T[0]) / dx
        k_eff_right_bound = enforce_sonic_limit(k_eff_right_bound, dT_dx_bound, Q_sonic, A_c)
        T_new[0] = T[0] + (dt / (rho * c_p * dx**2)) * (k_eff_right_bound * (T[1] - T[0]) -
                                                        k_eff_0 * (T[0] - T_ghost))

        # --- Right boundary (x = L_t) ---
        # Here, h_r is the linearized radiative coefficient:
        h_r = 4 * sigma * eps * (T_inf**3)
        T_ghost_end = ghost_node_condenser(T[-2], T[-1], dx, h_r, k_w, T_inf)
        region_end = get_region(x[-1], L_e, L_a, L_t)
        region_left_end = get_region(x[-2], L_e, L_a, L_t)
        k_eff_end = compute_k_eff(T[-1], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                h_lv, h_l, h_v, region_end)
        k_eff_prev = compute_k_eff(T[-2], P, R_v, mu_v, m_g, k_B, R_g, N_A,
                                h_lv, h_l, h_v, region_left_end)
        k_eff_left_bound = 0.5 * (k_eff_end + k_eff_prev)
        dT_dx_left_bound = (T[-1] - T[-2]) / dx
        k_eff_left_bound = enforce_sonic_limit(k_eff_left_bound, dT_dx_left_bound, Q_sonic, A_c)
        T_new[-1] = T[-1] + (dt / (rho * c_p * dx**2)) * (k_eff_end * (T_ghost_end - T[-1]) -
                                                        k_eff_left_bound * (T[-1] - T[-2]))

        T_history[step + 1, :] = T_new
        T = T_new.copy()
    
    return x, T_history