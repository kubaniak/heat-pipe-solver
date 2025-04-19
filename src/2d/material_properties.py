from fipy import CellVariable
from fipy.tools import numerix as npx

T_Na_crit = 2509.46  # K

def rho_sodium_s(T):
    return 972.70 - 0.2154 * (T - 273.15) 

def rho_sodium_l(T):
    return 219 + 275.32 * (1 - T / T_Na_crit) + 511.58 * (1 - T / T_Na_crit)**(0.5)

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

def c_p_sodium_s(T):
    return 1199 + 0.649 * (T - 273.15) + 1052.9e-5 * (T - 273.15)**2

def c_p_sodium_l(T):
    return 1436.72 - 0.58 * (T - 273.15) + 4.672e-4 * (T - 273.15)**2

def c_p_sodium_v(T): 
    """
    Interpolated with scipy and a polynomial of degree 6

    T_data: npx.arange(400, 2401, 100), 2500 is excluded because it is the critical point

    c_p_data: npx.array([
        860,
        1250,
        1800,
        2280,
        2590,
        2720,
        2700,
        2620,
        2510,
        2430,
        2390,
        2360,
        2340,
        2410,
        2460,
        2530,
        2660,
        2910,
        3400,
        4470,
        8030,
    ])

    Coefficients: [1.61051745e+04, -1.11332770e+02, 2.99491114e-01, -3.75559606e-04, 2.41879914e-07, -7.77357096e-11, 9.89238987e-15]
    """
    return  1.61051745e+04 + -1.11332770e+02 * T + 2.99491114e-01 + T**2 \
        + -3.75559606e-04 * T**3 + 2.41879914e-07 * T**4 + -7.77357096e-11 * T**5 + 9.89238987e-15 * T**6

def c_v_sodium_v(T):
    """
    Interpolated with scipy and a polynomial of degree 6

    T_data: npx.arange(400, 2401, 100), 2500 is excluded because it is the critical point

    c_v_data: npx.array([
        490,
        840,
        1310,
        1710,
        1930,
        1980,
        1920,
        1810,
        1680,
        1580,
        1510,
        1440,
        1390,
        1380,
        1360,
        1330,
        1300,
        1300,
        1340,
        1440,
        1760
    ])

    Coefficients: [2.94466965e+03, -2.73794530e+01,  8.93486407e-02, -1.17644463e-04, 7.52149483e-08, -2.33846884e-11,  2.84032804e-15]
    """
    return 2.94466965e+03 + -2.73794530e+01 * T + 8.93486407e-02 + T**2 \
        + -1.17644463e-04 * T**3 + 7.52149483e-08 * T**4 + -2.33846884e-11 * T**5 + 2.84032804e-15 * T**6

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
        'specific_heat_solid_pressure': c_p_sodium_s,
        'specific_heat_liquid_pressure': c_p_sodium_l,
        'specific_heat_vapor_pressure': c_p_sodium_v,
        'specific_heat_vapor_volume': c_v_sodium_v,
        'enthalpy_liquid': h_l_sodium,
        'enthalpy_vapor': h_v_sodium,
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
    """
    Intepolated with scipy and a linear function

    T_data: npx.arange(300, 1701, 100)

    c_p_data: npx.array([
        510.0296,
        523.4184,
        536.8072,
        550.196,
        564.0032,
        577.392,
        590.7808,
        604.1696,
        617.5584,
        631.3656,
        644.7544,
        658.1432,
        671.532,
        685.3392,
        698.728
    ])

    Coefficients: [0.13481446, 469.46671619]
    """
    return 0.1348144571428571 * T + 469.466716190476


def get_steel_properties():
    """
    Returns a dictionary of steel properties as functions of temperature.
    """
    return {
        'density': rho_steel,
        'thermal_conductivity': k_steel,
        'specific_heat': c_p_steel,
    }

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------

def get_k_Na_i(T: CellVariable, parameters: dict, k_l, k_s):
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']
    T_low = T_m - delta_T
    T_high = T_m + delta_T

    return npx.where(T < T_low, k_s,
           npx.where(T > T_high, k_l,
           k_s + (k_l - k_s) * ((T - T_low) / (2 * delta_T))))
    
def get_c_p_Na_i(T: CellVariable, sodium_properties:dict, parameters: dict, c_p_l, c_p_s):
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']
    T_low = T_m - delta_T
    T_high = T_m + delta_T
    h_vap = sodium_properties['heat_of_vaporization'](T)

    middle_value = (c_p_s + c_p_l) / 2 + h_vap / (2 * delta_T)

    return npx.where(T < T_low, c_p_s,
           npx.where(T > T_high, c_p_l, middle_value))

def get_rho_Na_i(T: CellVariable, parameters: dict, rho_l, rho_s):
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']
    T_low = T_m - delta_T
    T_high = T_m + delta_T

    interpolated = rho_s + (rho_l - rho_s) * ((T - T_low) / (2 * delta_T))

    return npx.where(T < T_low, rho_s,
           npx.where(T > T_high, rho_l, interpolated))

# --------------------------------------------------------
# Effective values for the vapor core region
# --------------------------------------------------------

def k_eff_vc(T: CellVariable, region: str, sodium_properties: dict, dimensions: dict, parameters: dict, constants: dict):
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


def get_vc_properties():
    """
    Returns a dictionary of effective properties for the vapor core region.
    """
    return {
        'thermal_conductivity': k_eff_vc,
        'specific_heat': c_p_sodium_v,
        'density': rho_sodium_v,
    }

# --------------------------------------------------------
# Effective values for the wick region
# --------------------------------------------------------

def k_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict):
    """
    Effective thermal conductivity for the wick region.
    """
    k_s = sodium_properties['thermal_conductivity_solid'](T)
    k_l = sodium_properties['thermal_conductivity_liquid'](T)

    k_steel = steel_properties['thermal_conductivity'](T)
    epsilon = parameters['porosity_wick']

    k_Na_i = get_k_Na_i(T, parameters, k_l, k_s)

    return k_Na_i * ((k_Na_i + k_steel) - (1 - epsilon) * (k_Na_i - k_steel)) / ((k_Na_i + k_steel) + (1 - epsilon) * (k_Na_i - k_steel))

def c_p_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict):
    """
    Effective specific heat for the wick region. 
    """
    c_p_l = sodium_properties['c_p_liquid'](T)
    c_p_s = sodium_properties['c_p_solid'](T)
    c_p_Na_i = get_c_p_Na_i(T, parameters, c_p_l, c_p_s)
    epsilon = parameters['porosity_wick']

    return epsilon * c_p_Na_i + (1 - epsilon) * steel_properties['specific_heat'](T)

def rho_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict):
    """
    Effective density for the wick region. 
    """
    rho_l = sodium_properties['density_liquid'](T)
    rho_s = sodium_properties['density_solid'](T)
    rho_Na_i = get_rho_Na_i(T, parameters, rho_l, rho_s)
    epsilon = parameters['porosity_wick']

    return epsilon * rho_Na_i + (1 - epsilon) * steel_properties['density'](T)


def get_wick_properties():
    """
    Returns a dictionary of effective properties for the wick region.
    """
    return {
        'thermal_conductivity': k_eff_wick,
        'specific_heat': c_p_eff_wick,
        'density': rho_eff_wick,
    }