import numpy as np

# Define a helper function that determines the region for a given axial position.
def get_region(x, L_e, L_a, L_t):
    """
    Returns a string indicating the region: 'evap_con' for evaporator/condenser or 'adiabatic'
    based on the axial coordinate x.
    L_e: evaporator length; L_a: adiabatic length; L_t = L_e + L_a + L_c.
    For simplicity, assume x in [0, L_e] and [L_e+L_a, L_t] use the same effective formulation.
    """
    if x <= L_e or x >= (L_e + L_a):
        return 'evap_con'
    else:
        return 'adiabatic'

def compute_D_vis(P, R_v, mu_v, section='evap_con'):
    """
    Compute the viscous diffusion coefficient D_vis.
    
    Parameters:
      P       : Saturated vapor pressure [Pa]
      R_v     : Vapor core radius [m]
      mu_v    : Dynamic viscosity of vapor [Pa·s]
      section : 'evap_con' for evaporator/condenser or 'adiabatic'
      
    Returns:
      D_vis   : Viscous diffusion coefficient [m²/s]
    """
    if section == 'evap_con':
        # For evaporator/condenser sections: D_vis = R_v^2 * mu_v / (4 * P)
        D_vis = (R_v**2 * mu_v) / (4 * P)
    elif section == 'adiabatic':
        # For adiabatic section: D_vis = R_v^2 * mu_v / (8 * P)
        D_vis = (R_v**2 * mu_v) / (8 * P)
    else:
        raise ValueError("Section must be either 'evap_con' or 'adiabatic'")
    return D_vis

def compute_D_K(T, R_v, m_g, k_B):
    """
    Compute the Knudsen diffusion coefficient D_K.
    
    Parameters:
      T   : Absolute temperature [K]
      R_v : Vapor core radius [m]
      m_g : Molecular mass of vapor [kg]
      k_B : Boltzmann constant [J/K]
      
    Returns:
      D_K : Knudsen diffusion coefficient [m²/s]
    """
    # Compute average molecular speed: c_bar = sqrt(8 * k_B * T / (pi * m_g))
    c_bar = np.sqrt(8 * k_B * T / (np.pi * m_g))
    # D_K = (2 * R_v / 3) * c_bar
    D_K = (2 * R_v / 3) * c_bar
    return D_K

def compute_D(P, R_v, mu_v, T, m_g, k_B, section='evap_con'):
    """
    Compute the total diffusion coefficient D = D_vis + D_K.
    
    Parameters:
      P       : Saturated vapor pressure [Pa]
      R_v     : Vapor core radius [m]
      mu_v    : Dynamic viscosity of vapor [Pa·s]
      T       : Absolute temperature [K]
      m_g     : Molecular mass of vapor [kg]
      k_B     : Boltzmann constant [J/K]
      section : 'evap_con' or 'adiabatic'
      
    Returns:
      D : Total diffusion coefficient [m²/s]
    """
    D_vis = compute_D_vis(P, R_v, mu_v, section)
    D_K = compute_D_K(T, R_v, m_g, k_B)
    D_total = D_vis + D_K
    return D_total

def compute_k_eff(T, P, R_v, mu_v, m_g, k_B, R_g, N_A, h_lv, 
                  h_l=None, h_v=None, section='evap_con'):
    """
    Compute the effective thermal conductivity k_eff for the vapor region.
    
    For evaporator/condenser sections, use h_l (latent heat "coefficient").
    For the adiabatic section, use h_v (vapor enthalpy).
    
    Parameters:
      T       : Temperature [K]
      P       : Saturated vapor pressure [Pa]
      R_v     : Vapor core radius [m]
      mu_v    : Dynamic viscosity of vapor [Pa·s]
      m_g     : Molecular mass of vapor [kg]
      k_B     : Boltzmann constant [J/K]
      R_g     : Specific gas constant [J/kg-K]
      N_A     : Avogadro's number [1/mol]
      h_lv    : Latent heat of vaporization [J/kg]
      h_l     : Latent heat (for evaporator/condenser) [J/kg] (use if section=='evap_con')
      h_v     : Vapor enthalpy [J/kg] (use if section=='adiabatic')
      section : 'evap_con' for evaporator/condenser sections or 'adiabatic'
      
    Returns:
      k_eff : Effective thermal conductivity [W/m-K]
    """
    # Compute total diffusion coefficient D
    D = compute_D(P, R_v, mu_v, T, m_g, k_B, section)
    
    # Choose the appropriate energy term depending on section.
    if section == 'evap_con':
        if h_l is None:
            raise ValueError("For evaporator/condenser section, parameter h_l must be provided.")
        energy_term = h_l
    elif section == 'adiabatic':
        if h_v is None:
            raise ValueError("For adiabatic section, parameter h_v must be provided.")
        energy_term = h_v
    else:
        raise ValueError("Section must be either 'evap_con' or 'adiabatic'")
    
    # Compute k_eff using the derived formula:
    # k_eff = - (D * energy_term * h_lv * M_g * P) / (R_g * N_A * k_B * T^3)
    # Note: M_g is the molar mass of the vapor. We assume it is provided as part of h_lv if needed.
    # For clarity, we assume M_g is embedded in the energy term or provided separately.
    # Here, we assume M_g is included as a parameter (and thus the caller must supply it).
    M_g = 0.0229897693  # <-- Insert the molar mass of vapor [kg/mol] here or pass it as a parameter.
    if M_g is None:
        # If no value is provided, leave it as a placeholder.
        raise ValueError("Molar mass M_g must be provided. Please update the code with its value.")
    
    # Return the absolute value of k_eff for physical consistency with heat flow direction
    k_eff = abs((D * energy_term * h_lv * M_g * P) / (R_g * N_A * k_B * (T**3)))
    return k_eff

# Example usage (with dummy values):
if __name__ == "__main__":
    # Dummy parameters (replace with actual values as needed)
    T = 700.0                # Temperature [K]
    P = 1e5                  # Saturated vapor pressure [Pa]
    R_v = 0.01               # Vapor core radius [m]
    mu_v = 1e-5              # Dynamic viscosity [Pa·s]
    m_g = 4.65e-26           # Molecular mass [kg] (example: sodium atom ~ 3.8e-26 kg, adjust accordingly)
    k_B = 1.380649e-23       # Boltzmann constant [J/K]
    R_g = 461.5              # Specific gas constant [J/kg-K] (example value, adjust for vapor)
    N_A = 6.022e23           # Avogadro's number [1/mol]
    h_lv = 2.26e6            # Latent heat of vaporization [J/kg] (example value)
    h_l = 2.0e6              # Latent heat "coefficient" for evaporator/condenser [J/kg] (example value)
    h_v = 1.8e6              # Vapor enthalpy for adiabatic section [J/kg] (example value)
    M_g = 3.8e-26            # Molar mass of vapor [kg/mol] (example value for sodium, adjust accordingly)
    
    # Compute effective thermal conductivity for evaporator/condenser section
    k_eff_evap = compute_k_eff(T, P, R_v, mu_v, m_g, k_B, R_g, N_A, h_lv, h_l, h_v, section='evap_con', M_g_val=M_g)
    print("Effective thermal conductivity (evaporator/condenser):", k_eff_evap, "W/m-K")
    
    # Compute effective thermal conductivity for adiabatic section
    k_eff_adiab = compute_k_eff(T, P, R_v, mu_v, m_g, k_B, R_g, N_A, h_lv, h_l, h_v, section='adiabatic', M_g_val=M_g)
    print("Effective thermal conductivity (adiabatic):", k_eff_adiab, "W/m-K")
