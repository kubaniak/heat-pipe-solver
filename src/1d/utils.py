# Define a function to compute the sonic limit effective conductivity.
def enforce_sonic_limit(k_eff_computed, dT_dx, Q_sonic, A_c, eps=1e-8):
    """
    Enforce the sonic limit on the effective conductivity.
    
    k_eff_computed: the computed effective conductivity from the conduction model [W/m-K]
    dT_dx: local temperature gradient [K/m]
    Q_sonic: sonic heat flux limit [W]
    A_c: vapor core cross-sectional area [m^2]
    eps: small number to avoid division by zero
    
    Returns: k_eff_limited such that
        k_eff <= Q_sonic / (A_c * |dT_dx|)
    """
    # Avoid division by zero:
    grad = abs(dT_dx) if abs(dT_dx) > eps else eps
    k_limit = Q_sonic / (A_c * grad)
    # if k_eff_computed > k_limit:
    #     print(f"Enforcing sonic limit: k_eff = {k_eff_computed:.2f} W/m-K, k_limit = {k_limit:.2f} W/m-K")  
    return min(k_eff_computed, k_limit)

def ghost_node_evaporator(T1, dx, q_e, k_w):
    """Compute ghost node for evaporator boundary."""
    return T1 + 2 * dx * q_e / k_w

def ghost_node_condenser_nonlinear(T_prev, T_last, dx, sigma, eps, k_w, T_inf):
    """
    Compute the ghost node value at the condenser (x = L_t + dx)
    using the full nonlinear radiative boundary condition:
      -k_w*(T_ghost - T_prev)/(2*dx) = sigma*eps*(T_last**4 - T_inf**4)
    Solved for T_ghost.
    """
    return T_prev - (2 * dx * sigma * eps / k_w) * (T_last**4 - T_inf**4)