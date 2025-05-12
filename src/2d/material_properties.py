from fipy import CellVariable
from fipy.tools import numerix as npx

# --- Constants ---
T_Na_crit = 2509.46  # K
delta_H_melt = 113e3  # J/kg

# --- Sodium: Solid, Liquid, Vapor Properties ---
def rho_sodium_s(T): return 972.70 - 0.2154 * (T - 273.15)
def rho_sodium_l(T): return 219 + 275.32 * (1 - T / T_Na_crit) + 511.58 * (1 - T / T_Na_crit)**0.5

def h_vap_sodium(T):
    return (393.37 * (1 - T / T_Na_crit) + 4398.6 * (1 - T / T_Na_crit)**0.29302) * 1e3

def dPdT_sodium(T):
    return (12633.73 / T**2 - 0.4672 / T) * npx.exp(11.9463 - 12633.73 / T - 0.4672 * npx.log(T)) * 1e6

def rho_sodium_v(T):
    return (h_vap_sodium(T) / (T * dPdT_sodium(T)) + 1 / rho_sodium_l(T))**-1

def k_sodium_s(T): return 135.6 - 0.167 * (T - 273.15)
def k_sodium_l(T): return 124.67 - 0.11381 * T + 5.5226e-5 * T**2 - 1.1842e-8 * T**3

def c_p_sodium_s(T): return 1199 + 0.649 * (T - 273.15) + 1052.9e-5 * (T - 273.15)**2
def c_p_sodium_l(T): return 1436.72 - 0.58 * (T - 273.15) + 4.672e-4 * (T - 273.15)**2

def c_p_sodium_v(T):
    return (16105.174483770606 - 111.33277032117233 * T + 0.2994911138258808 * T**2 - 0.00037555960577985903 * T**3
            + 2.418799137943085e-07 * T**4 - 7.773570963102343e-11 * T**5 + 2.8698379132831126e-15 * T**6)

def c_v_sodium_v(T):
    return (2944.6696506593726 - 27.379453000849395 * T + 0.08934864074702555 * T**2 - 0.00011764446335597157 * T**3\
             + 7.521494832758196e-08 * T**4 - 2.338468836342328e-11 * T**5 + 9.892389871569863e-15 * T**6)

def mu_sodium_v(T): return 6.083e-9 * T + 1.2606e-5
def P_sat_sodium(T): return npx.exp(11.9463 - 12633.73 / T - 0.4672 * npx.log(T)) * 1e6

def h_l_sodium(T):
    return (-365.77 + 1.6582 * T - 4.2395e-4 * T**2 + 1.4847e-7 * T**3 + 2992.6 * T**-1) * 1e3

def h_v_sodium(T): return h_l_sodium(T) + h_vap_sodium(T)

def get_sodium_properties():
    return {
        'density_solid': rho_sodium_s,
        'density_liquid': rho_sodium_l,
        'density_vapor': rho_sodium_v,
        'heat_of_vaporization': h_vap_sodium,
        'thermal_conductivity_solid': k_sodium_s,
        'thermal_conductivity_liquid': k_sodium_l,
        'specific_heat_solid': c_p_sodium_s,
        'specific_heat_liquid': c_p_sodium_l,
        'specific_heat_vapor_pressure': c_p_sodium_v,
        'specific_heat_vapor_volume': c_v_sodium_v,
        'enthalpy_liquid': h_l_sodium,
        'enthalpy_vapor': h_v_sodium,
        'viscosity': mu_sodium_v,
        'vapor_pressure': P_sat_sodium,
    }

# --- Steel Properties ---
def rho_steel(T): return (7.9841 - 2.656e-4 * T - 1.158e-7 * T**2) * 1e3
def k_steel(T): return (8.116e-2 + 1.618e-4 * T) * 100
def c_p_steel(T): return 0.1348 * T + 469.47

def rho_eff_steel(T): return rho_steel(T) + 3.75e6 / c_p_steel(T)

def get_steel_properties():
    return {
        'density_evap_adia': rho_eff_steel,
        'density_cond': rho_steel,
        'thermal_conductivity': k_steel,
        'specific_heat': c_p_steel,
    }

# --- Interpolated Sodium Properties ---
def _interp_linear(T, T_low, T_high, val_low, val_high):
    return val_low + (val_high - val_low) * ((T - T_low) / (T_high - T_low))

def get_k_Na_i(T, params, k_l, k_s):
    T_m, dT = params['T_melting_sodium'], params['delta_T_sodium']
    return k_s * (T < T_m - dT) + k_l * (T > T_m + dT) + _interp_linear(T, T_m - dT, T_m + dT, k_s, k_l) * ((T >= T_m - dT) & (T <= T_m + dT))

def get_c_p_Na_i(T, sodium_props, params, c_p_l, c_p_s):
    T_m, dT = params['T_melting_sodium'], params['delta_T_sodium']
    h_melt = delta_H_melt
    mid_val = (c_p_s + c_p_l) / 2 + h_melt / (2 * dT)
    return c_p_s * (T < T_m - dT) + c_p_l * (T > T_m + dT) + mid_val * ((T >= T_m - dT) & (T <= T_m + dT))

def get_rho_Na_i(T, params, rho_l, rho_s):
    T_m, dT = params['T_melting_sodium'], params['delta_T_sodium']
    interp = _interp_linear(T, T_m - dT, T_m + dT, rho_s, rho_l)
    return rho_s * (T < T_m - dT) + rho_l * (T > T_m + dT) + interp * ((T >= T_m - dT) & (T <= T_m + dT))

# --- Vapor Core Effective Properties ---
def get_Q_sonic(T, end_cap_mask, mesh, sodium_props, dims, consts):
    """
    Calculate the sonic heat transfer rate in the vapor core region.
    """
    T_v_0 = npx.sum(T * end_cap_mask) / npx.sum(end_cap_mask)
    rho_v_0 = sodium_props['density_vapor'](T_v_0) # density at the evaporator end cap
    A_vc = npx.pi * dims['R_vc']**2
    h_vap = sodium_props['heat_of_vaporization'](T)
    gamma = sodium_props['specific_heat_vapor_pressure'](T) / sodium_props['specific_heat_vapor_volume'](T)
    R_g = consts['R']

    Q = CellVariable(mesh=mesh, 
                     value=(rho_v_0 * A_vc * h_vap * npx.sqrt(gamma * R_g * T_v_0)) / (npx.sqrt(2 * (gamma + 1))),
                     name='Q_sonic')

    return Q

def k_eff_vc(T, end_cap_mask, mesh, region, sodium_props, dims, params, consts):
    R_v = dims['R_vc']
    mu = sodium_props['viscosity'](T)
    P = sodium_props['vapor_pressure'](T)
    R = consts['R']
    M = params['M_g_sodium']
    h_l = sodium_props['enthalpy_liquid'](T)
    h_v = sodium_props['enthalpy_vapor'](T)
    h_vap = sodium_props['heat_of_vaporization'](T)
    factor = (2 * R_v / 3) * npx.sqrt((8 * R * T) / (npx.pi * M))
    base = (R_v**2 * P) / (4 if region == 'evap_cond' else 8) / mu + factor
    enthalpy = h_l if region == 'evap_cond' else h_v

    k_eff = base * (enthalpy * h_vap * M**2 * P) / (R**2 * T**3)
    return k_eff
    # --- Sonic limit enforcement ---
    # Q_sonic is a global limit, but we apply it locally for each cell
    # Assume axial direction is x (index 0)
    eps = 1e-12
    A_c = npx.pi * R_v**2
    # Compute |∇T_v,axial| (FiPy: T.grad[0]) (numpy: npx.gradient(T)[0])
    grad_T_axial = npx.abs(T.grad[0]) + eps
    # grad_T_axial = npx.gradient(T)[0] + eps
    # Q_sonic is a CellVariable
    Q_sonic = get_Q_sonic(T, end_cap_mask, mesh, sodium_props, dims, consts)
    # k_limit is a CellVariable
    k_limit = Q_sonic / (A_c * grad_T_axial)
    # Enforce the limit
    k_eff_limited = k_eff * (k_eff <= k_limit) + k_limit * (k_eff > k_limit)
    return k_eff_limited

    

def get_vc_properties():
    return {
        'thermal_conductivity': k_eff_vc,
        'specific_heat': c_p_sodium_v,
        'density': rho_sodium_v,
    }

# --- Wick Effective Properties ---
def k_eff_wick(T, Na_props, steel_props, params):
    k_s = Na_props['thermal_conductivity_solid'](T)
    k_l = Na_props['thermal_conductivity_liquid'](T)
    k_steel = steel_props['thermal_conductivity'](T)
    epsilon = params['porosity_wick']
    k_Na_i = get_k_Na_i(T, params, k_l, k_s)

    numerator = (k_Na_i + k_steel) - (1 - epsilon) * (k_Na_i - k_steel)
    denominator = (k_Na_i + k_steel) + (1 - epsilon) * (k_Na_i - k_steel)

    return k_Na_i * numerator / denominator

def c_p_eff_wick(T, Na_props, steel_props, params):
    c_p_l = Na_props['specific_heat_liquid'](T)
    c_p_s = Na_props['specific_heat_solid'](T)
    c_p_Na_i = get_c_p_Na_i(T, Na_props, params, c_p_l, c_p_s)
    epsilon = params['porosity_wick']
    return epsilon * c_p_Na_i + (1 - epsilon) * steel_props['specific_heat'](T)

def rho_eff_wick(T, Na_props, steel_props, params):
    rho_l = Na_props['density_liquid'](T)
    rho_s = Na_props['density_solid'](T)
    rho_Na_i = get_rho_Na_i(T, params, rho_l, rho_s)
    epsilon = params['porosity_wick']
    return epsilon * rho_Na_i + (1 - epsilon) * steel_props['density_cond'](T)

def get_wick_properties():
    return {
        'thermal_conductivity': k_eff_wick,
        'specific_heat': c_p_eff_wick,
        'density': rho_eff_wick,
    }
