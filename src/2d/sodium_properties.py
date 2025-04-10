import numpy as np
from scipy.interpolate import interp1d
from typing import Union

def get_density(state: str = 'vapor') -> interp1d:
    """
    Return interpolation function for sodium density as a function of temperature.
    
    Args:
        state: 'liquid' or 'vapor' to specify the state of sodium
        
    Returns:
        Interpolation function that takes temperature in K and returns density in kg/m³
    """
    # Data points for sodium density (kg/m³) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100) 
    
    # Density data for liquid sodium (kg/m³)
    rho_data_liquid = np.array([920.7, 897.3, 873.6, 849.8, 825.8, 801.7, 777.5, 753.3, 729.1, 704.9, 680.7, 656.6, 632.7, 608.9, 585.1, 561.1, 536.4, 510.3, 481.7, 448.1, 403.2, 300.9])

    # Density data for sodium vapor (kg/m³)
    rho_data_vapor = np.array([9.601e-10, 4.34e-7, 2.409e-5, 4.083e-4, 3.321e-3, 1.662e-2, 5.938e-2, 0.1663, 0.3889, 0.7928, 1.454, 2.455, 3.881, 5.820, 8.365, 11.63, 15.75, 20.96, 27.65, 36.71, 50.89, 99.75])

    if state == 'liquid':
        rho_data = rho_data_liquid
    else:
        rho_data = rho_data_vapor

    # Interpolate the density data
    density_interp = interp1d(T_data, rho_data, kind='cubic', fill_value='extrapolate')
    return density_interp

def get_specific_heat(state: str = 'vapor') -> interp1d:
    """
    Return interpolation function for sodium specific heat at constant volume as a function of temperature.
    
    Args:
        state: 'liquid' or 'vapor' to specify the state of sodium
        
    Returns:
        Interpolation function that takes temperature in K and returns specific heat in J/kg-K
    """
    # Data points for sodium specific heat (J/kg-K) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Specific heat data for liquid sodium (J/kg-K)
    c_p_data_liquid = np.array([28.28, 26.46, 24.94, 23.66, 22.59, 21.71, 21.03, 20.54, 20.22, 20.07, 20.10, 20.30, 20.69, 21.27, 21.99, 22.88, 23.99, 25.45, 27.51, 30.91, 38.69, 135.8])
    
    # Specific heat data for sodium vapor (J/kg-K)
    c_p_data_vapor = np.array([15.16, 21.14, 29.45, 36.93, 41.57, 43.19, 42.66, 41.01, 38.93, 36.83, 34.83, 32.93, 31.07, 29.28, 28.05, 27.42, 27.39, 28.07, 29.85, 34.03, 47.20, 366.8])

    if state == 'liquid':
        c_p_data = c_p_data_liquid
    else:
        c_p_data = c_p_data_vapor

    # Interpolate the specific heat data
    specific_heat_interp = interp1d(T_data, c_p_data, kind='cubic', fill_value='extrapolate')
    return specific_heat_interp

def get_vapor_pressure() -> interp1d:
    """
    Return interpolation function for sodium vapor pressure as a function of temperature.
    
    Returns:
        Interpolation function that takes temperature in K and returns vapor pressure in Pa
    """
    # Data points for sodium vapor pressure (MPa) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Vapor pressure data for sodium (MPa)
    P_data = np.array([1.358e-10, 7.635e-8, 5.047e-6, 9.869e-5, 9.043e-4, 5.010e-3, 1.955e-2, 5.917e-2, 0.1482, 0.3209, 0.6203, 1.096, 1.798, 2.779, 4.087, 5.766, 7.851, 10.37, 13.36, 16.81, 20.76, 25.19])
    P_data = P_data * 1e6  # Convert from MPa to Pa

    # Interpolate the vapor pressure data
    vapor_pressure_interp = interp1d(T_data, P_data, kind='cubic', fill_value='extrapolate')
    return vapor_pressure_interp

def get_viscosity() -> interp1d:
    """
    Return interpolation function for sodium dynamic viscosity as a function of temperature.
    
    Returns:
        Interpolation function that takes temperature in K and returns dynamic viscosity in Pa-s
    """
    # Data points for sodium dynamic viscosity (Pa-s * 10^3) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Dynamic viscosity data for sodium (Pa-s * 10^3)
    mu_data = np.array([
        0.614, 0.416, 0.320, 0.265, 0.229, 0.204, 0.185, 0.171, 0.160, 0.150, 
        0.142, 0.136, 0.130, 0.125, 0.120, 0.116, 0.112, 0.108, 0.105, 0.102, 
        0.0992, 0.0965
    ]) * 1e-3  # Convert to Pa-s

    # Interpolate the viscosity data
    viscosity_interp = interp1d(T_data, mu_data, kind='cubic', fill_value='extrapolate')
    return viscosity_interp

def get_heat_of_vaporization() -> interp1d:
    """
    Return interpolation function for sodium heat of vaporization as a function of temperature.
    
    Returns:
        Interpolation function that takes temperature in K and returns heat of vaporization in J/kg
    """
    # Data points for sodium heat of vaporization (J/kg) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Heat of vaporization data for sodium (J/kg)
    h_lv_data = np.array([103.3, 102.1, 100.6, 98.78, 96.77, 94.63, 92.47, 
                          90.32, 88.17, 85.98, 83.68, 81.21, 78.53, 75.64, 
                          72.49, 69.03, 65.15, 60.72, 55.48, 48.93, 39.69, 18.03])
    
    # Interpolate the heat of vaporization data
    heat_of_vaporization_interp = interp1d(T_data, h_lv_data, kind='cubic', fill_value='extrapolate')
    return heat_of_vaporization_interp

def get_enthalpy(state: str = 'vapor') -> interp1d:
    """
    Return interpolation function for sodium enthalpy as a function of temperature.
    
    Args:
        state: 'liquid' or 'vapor' to specify the state of sodium
        
    Returns:
        Interpolation function that takes temperature in K and returns enthalpy in J/kg
    """
    # Data points for sodium enthalpy (J/kg) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Enthalpy data for sodium vapor (J/kg)
    h_v_data = np.array([109.0, 110.8, 112.4, 113.5, 114.4, 115.2, 115.9, 116.7, 117.5, 118.2, 119.0, 119.6, 120.1, 120.5, 120.8, 120.9, 120.8, 120.5, 119.7, 118.3, 115.5, 106.6])

    # Enthalpy data for sodium liquid (J/kg)
    h_l_data = np.array([5.669, 8.774, 11.79, 14.75, 17.67, 20.57, 23.46, 26.36, 29.29, 32.26, 35.30, 38.40, 41.59, 44.89, 48.32, 51.90, 55.69, 59.76, 64.23, 69.36, 75.83, 88.52])
    
    if state == 'liquid':
        enthalpy_data = h_l_data
    else:
        enthalpy_data = h_v_data

    # Interpolate the enthalpy data
    enthalpy_interp = interp1d(T_data, enthalpy_data, kind='cubic', fill_value='extrapolate')
    return enthalpy_interp

def get_thermal_conductivity() -> interp1d:
    """
    Return interpolation function for sodium thermal conductivity as a function of temperature.
    
    Returns:
        Interpolation function that takes temperature in K and returns thermal conductivity in W/m-K
    """
    # Data points for sodium thermal conductivity (W/m-K) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Thermal conductivity data for sodium (W/m-K)
    k_data = np.array([86.8, 81.8, 76.4, 71.5, 66.8, 62.5, 58.3, 54.3, 50.3, 46.5, 42.2, 39.3, 35.8, 32.3, 28.9, 25.5, 22.2, 18.9, 15.6, 12.3, 9.5, 5.8])

    # Interpolate the thermal conductivity data
    thermal_conductivity_interp = interp1d(T_data, k_data, kind='cubic', fill_value='extrapolate')
    return thermal_conductivity_interp

def get_sodium_properties(state: str = 'vapor'):
    """
    Return a dictionary of interpolation functions for sodium properties.
    
    Args:
        state: 'liquid' or 'vapor' to specify the state of sodium
        
    Returns:
        Dictionary of interpolation functions for sodium properties:
        density - Density (kg/m³)
        specific_heat - Specific heat (J/kg-K)
        vapor_pressure - Vapor pressure (Pa)
        viscosity - Dynamic viscosity (Pa-s)
        heat_of_vaporization - Heat of vaporization (J/kg)
        enthalpy - Enthalpy (J/kg)
        thermal_conductivity - Thermal conductivity (W/m-K)
    """
    return {
        "density": get_density(state),
        "specific_heat": get_specific_heat(state),
        "vapor_pressure": get_vapor_pressure(),
        "viscosity": get_viscosity(),
        "heat_of_vaporization": get_heat_of_vaporization(),
        "enthalpy": get_enthalpy(state),
        "thermal_conductivity": get_thermal_conductivity()
    }

# Testing
if __name__ == '__main__':
    sodium_properties = get_sodium_properties()
    T = 420
    print(f"Sodium density at {T} K: {sodium_properties['density'](T)} kg/m³")
    print(f"Sodium specific heat at {T} K: {sodium_properties['specific_heat'](T)} J/kg-K")
    print(f"Sodium vapor pressure at {T} K: {sodium_properties['vapor_pressure'](T)} Pa")
    print(f"Sodium dynamic viscosity at {T} K: {sodium_properties['viscosity'](T)} Pa-s")
    print(f"Sodium heat of vaporization at {T} K: {sodium_properties['heat_of_vaporization'](T)} J/kg")
    print(f"Sodium enthalpy at {T} K: {sodium_properties['enthalpy'](T)} J/kg")
    print(f"Sodium thermal conductivity at {T} K: {sodium_properties['thermal_conductivity'](T)} W/m-K")