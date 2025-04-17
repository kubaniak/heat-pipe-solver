from fipy.tools import print_function
from fipy import CellVariable
from fipy.tools import numerix as npx

T_Na_crit = 2509.46  # K

def rho_sodium_s(T):
    return 972.70 - 0.2154 * (T - 273.15) 

def rho_sodium_l(T):
    return 219 + 28^75.32 * (1 - T / T_Na_crit) + 511.58 * (1 - T / T_Na_crit)**(0.5)

def h_vap_sodium(T):
    return (393.37 * (1 - T / T_Na_crit) + 4398.6 * (1 - T / T_Na_crit)**(0.28302)) * 1e3 

def dPdT_sodium(T):
    return (12633.73 / T**2 - 0.4672 / T) * npx.exp(11.9463 - 12633.73 / T - 0.4672 * npx.log(T)) # MPa/K

def rho_sodium_v(T):
    return (h_vap_sodium(T) / (T * dPdT_sodium(T)) + 1 / rho_sodium_l(T)) ** -1

def k_sodium_s(T):
    return 135.6 - 0.167 * (T - 273.15) 

def k_sodium_l(T):
    return 124.67 - 0.11381 * T + 5.5226e-5 * T**2 - 1.1842e-8 * T**3

def k_sodium_v(T):
    return 0 # THIS SHOULD BE k_eff_v(T, sodium_properties, steel_properties, parameters) TODO

def c_p_sodium_s(T):
    return 1199 + 0.649 * (T - 273.15) + 1052.9e-5 * (T - 273.15)**2

def c_p_sodium_l(T):
    return 1436.72 - 0.58 * (T - 273.15) + 4.672e-4 * (T - 273.15)**2

def c_p_sodium_v(T):
    return 0 # THIS SHOULD BE DONE WITH TABLE INTERPOLATION, but in symbolic form! TODO

def mu_sodium_v(T):
    return 6.083e-9 * T + 1.2606e-5

def P_sat_sodium(T):
    return npx.exp(11.9463 - 12633.73 / T - 0.4672 * npx.log(T)) # MPa

def h_l_sodium(T):
    return (-365.77 + 1.6582 * T - 4.2395e-4 * T**2 + 1.4847e-7 * T**3 + 2992.6 * T**-1) * 1e3

def h_v_sodium(T):
    return (h_l_sodium(T) + h_vap_sodium(T))


def get_sodium_properties():
    """
    Returns a dictionary of sodium properties as functions of temperature.
    """
    return {
        'density_solid': rho_sodium_s,
        'density_liquid': rho_sodium_l,
        'density_vapor': rho_sodium_v,
        'heat_of_vaporization': h_vap_sodium,
        'thermal_conductivity_solid': k_sodium_s,
        'thermal_conductivity_liquid': k_sodium_l,
        'thermal_conductivity_vapor': k_sodium_v,
        'specific_heat_solid': c_p_sodium_s,
        'specific_heat_liquid': c_p_sodium_l,
        'specific_heat_vapor': c_p_sodium_v,
        'viscosity': mu_sodium_v, 
        'vapor_pressure': P_sat_sodium,
    }

# --------------------------------------------------------
# Steel properties
# --------------------------------------------------------

def rho_steel(T):
    return (7.9841 - 2.6560e-4 * T - 1.158e-7 * T**2) * 1e3 # kg/m^3

def k_steel(T):
    return (8.116e-2 + 1.618e-4 * T) * 100  # W/m/K

def c_p_steel(T):
    return 0 # THIS SHOULD BE DONE WITH TABLE INTERPOLATION, but in symbolic form! TODO

# --------------------------------------------------------
# Effective values for the wick region
# --------------------------------------------------------

def get_k_Na_i(T: CellVariable, parameters: dict, k_l: float, k_s: float) -> float:
    """
    Effective thermal conductivity for sodium inside the wick region. Melting point is considered.
    """
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']

    if T < T_m - delta_T:
        return k_s
    elif T_m - delta_T <= T <= T_m + delta_T:
        return k_s + (k_l - k_s) * ((T - (T_m - delta_T)) / (2 * delta_T))
    else:
        return k_l
    
def get_c_p_Na_i(T: CellVariable, sodium_properties:dict, parameters: dict, c_p_l: float, c_p_s: float) -> float:
    """
    Effective specific heat for sodium inside the wick region. Melting point is considered.
    """
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']
    h_vap = sodium_properties['heat_of_vaporization'](T)

    if T < T_m - delta_T:
        return c_p_s
    elif T_m - delta_T <= T <= T_m + delta_T:
        return (c_p_s + c_p_l) / 2 + h_vap / (2 * delta_T)
    else:
        return c_p_l

def get_rho_Na_i(T: CellVariable, parameters: dict, rho_l: float, rho_s: float) -> float:
    """
    Effective density for sodium inside the wick region. Melting point is considered.
    """
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']

    if T < T_m - delta_T:
        return rho_s
    elif T_m - delta_T <= T <= T_m + delta_T:
        return rho_s + (rho_l - rho_s) * ((T - (T_m - delta_T)) / (2 * delta_T))
    else:
        return rho_l

def k_eff_vc(T: CellVariable, region: str, sodium_properties: dict, dimensions: dict, parameters: dict, constants: dict) -> float:
    """
    Effective thermal conductivity for the vapor core region.
    
    Args:
        T: Temperature in K
        region: 'evap_cond', adiabatic'

    Returns:
        Effective thermal conductivity in W/m-K
    """
    R_v = dimensions['R_vc']
    mu_v = sodium_properties['viscosity'](T)
    P = sodium_properties['vapor_pressure'](T)
    R = constants['R']
    M_g = parameters['M_g_sodium']
    h_l = sodium_properties['enthalpy_liquid'](T)
    h_v = sodium_properties['enthalpy_vapor'](T)
    h_vap = sodium_properties['heat_of_vaporization'](T)
    
    if region == 'evap_cond':
        res = -((R_v**2 * P) / (4 * mu_v) + (2 * R_v) / 3 * npx.sqrt((8 * R * T) / (npx.pi * M_g))) * (h_l * h_vap * M_g**2 * P) / (R**2 * T**3)
    elif region == 'adiabatic':
        res = -((R_v**2 * P) / (8 * mu_v) + (2 * R_v) / 3 * npx.sqrt((8 * R * T) / (npx.pi * M_g))) * (h_v * h_vap * M_g**2 * P) / (R**2 * T**3)
    else:
        raise ValueError("Invalid region specified. Use 'evap_cond' or 'adiabatic'.")
    
    return res

def k_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict) -> float:
    """
    Effective thermal conductivity for the wick region.
    """
    k_s = parameters['k_solid_sodium']
    k_l = sodium_properties['thermal_conductivity'](T)

    k_steel = steel_properties['thermal_conductivity'](T)
    epsilon = parameters['porosity_wick']

    k_Na_i = get_k_Na_i(T, parameters, k_l, k_s)

    return k_Na_i * ((k_Na_i + k_steel) - (1 - epsilon) * (k_Na_i - k_steel)) / ((k_Na_i + k_steel) + (1 - epsilon) * (k_Na_i - k_steel))

def c_p_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict) -> float:
    """
    Effective specific heat for the wick region. 
    """
    c_p_l = sodium_properties['c_p_liquid'](T)
    c_p_s = parameters['c_p_solid_sodium']
    c_p_Na_i = get_c_p_Na_i(T, parameters, c_p_l, c_p_s)
    epsilon = parameters['porosity_wick']

    return epsilon * c_p_Na_i + (1 - epsilon) * steel_properties['specific_heat'](T)

def rho_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict) -> float:
    """
    Effective density for the wick region. 
    """
    rho_l = sodium_properties['density_liquid'](T)
    rho_s = parameters['rho_solid_sodium']
    rho_Na_i = rho_Na_i(T, parameters, rho_l, rho_s)
    epsilon = parameters['porosity_wick']

    return epsilon * rho_Na_i + (1 - epsilon) * steel_properties['density'](T)