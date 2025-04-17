"""
material_properties.py
This module contains functions to compute material properties of the heat pipe compononents
as a function of temperature. 
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Union

from fipy import CellVariable
from fipy.tools import numerix as npx

# ----------------------------------------
# Material Properties of Sodium
# ----------------------------------------

import numpy as np
from scipy.interpolate import interp1d

def sodium_viscosity() -> interp1d:
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

def sodium_density(state: str = 'vapor') -> interp1d:
    """
    Return interpolation function for sodium density as a function of temperature. **USE THIS!** USAGE REGION: vapor core.
    
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

def sodium_specific_heat(state: str = 'vapor', constant='pressure') -> interp1d:
    """
    Return interpolation function for sodium specific heat as a function of temperature. 
    
    Args:
        state: 'liquid' or 'vapor' to specify the state of sodium
        constant: 'pressure' or 'volume' to specify the type of specific heat
        
    Returns:
        Interpolation function that takes temperature in K and returns specific heat in J/kg-K
    """
    # Data points for sodium specific heat (J/kg-K) as a function of temperature (K)
    T_data = np.arange(400, 2501, 100)

    # Specific heat data for liquid sodium (J/kg-K)
    c_p_data_liquid = np.array([31.58, 30.58, 29.87, 29.37, 29.05, 28.91, 28.94, 29.12, 29.45, 29.94, 30.58, 31.37, 32.32, 33.44, 34.85, 36.67, 39.14, 42.70, 48.30, 58.60, 84.75, 482.9])
    c_v_data_liquid = np.array([28.28, 26.46, 24.94, 23.66, 22.59, 21.71, 21.03, 20.54, 20.22, 20.07, 20.10, 20.30, 20.69, 21.27, 21.99, 22.88, 23.99, 25.45, 27.51, 30.91, 38.69, 135.8])
    
    # Specific heat data for sodium vapor (J/kg-K)
    c_p_data_vapor = np.array([23.52, 30.27, 40.12, 49.66, 56.38, 59.70, 60.32, 59.37, 57.81, 56.26, 54.97, 53.92, 52.90, 51.88, 51.98, 53.55, 57.05, 63.60, 76.02, 103.4, 194.9, 4123.0])
    c_v_data_vapor = np.array([15.16, 21.14, 29.45, 36.93, 41.57, 43.19, 42.66, 41.01, 38.93, 36.83, 34.83, 32.93, 31.07, 29.28, 28.05, 27.42, 27.39, 28.07, 29.85, 34.03, 47.20, 366.8])

    if state == 'liquid':
        if constant == 'pressure':
            c_data = c_p_data_liquid
        else:
            c_data = c_v_data_liquid
    else:
        if constant == 'pressure':
            c_data = c_p_data_vapor
        else:
            c_data = c_v_data_vapor

    # Interpolate the specific heat data
    specific_heat_interp = interp1d(T_data, c_data, kind='cubic', fill_value='extrapolate')
    return specific_heat_interp

def sodium_vapor_pressure() -> interp1d:
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

def sodium_heat_of_vaporization() -> interp1d:
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

def sodium_enthalpy(state: str = 'vapor') -> interp1d:
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

def sodium_thermal_conductivity() -> interp1d:
    """
    Return interpolation function for liquid sodium thermal conductivity as a function of temperature.
    
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

def get_sodium_properties() -> dict:
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
        "density_liquid": sodium_density('liquid'),
        "density_vapor": sodium_density('vapor'),
        "c_p_liquid": sodium_specific_heat('liquid', 'pressure'),
        "c_p_vapor": sodium_specific_heat('vapor', 'pressure'),
        "c_v_liquid": sodium_specific_heat('liquid', 'volume'),
        "c_v_vapor": sodium_specific_heat('vapor', 'volume'),
        "vapor_pressure": sodium_vapor_pressure(),
        "viscosity": sodium_viscosity(),
        "heat_of_vaporization": sodium_heat_of_vaporization(),
        "enthalpy_liquid": sodium_enthalpy('liquid'),
        "enthalpy_vapor": sodium_enthalpy('vapor'),
        "thermal_conductivity": sodium_thermal_conductivity()
    }

def get_tabulated_data() -> dict:
    """
    Return tabulated data for sodium properties as a function of temperature. Temperature range: 400 K to 2500 K in 100 K intervals.
    """
    return {
        "density_liquid": np.array([920.7, 897.3, 873.6, 849.8, 825.8, 801.7, 777.5, 753.3, 729.1, 704.9, 680.7, 656.6, 632.7, 608.9, 585.1, 561.1, 536.4, 510.3, 481.7, 448.1, 403.2, 300.9]),
        "density_vapor": np.array([9.601e-10, 4.34e-7, 2.409e-5, 4.083e-4, 3.321e-3, 1.662e-2, 5.938e-2, 0.1663, 0.3889, 0.7928, 1.454, 2.455, 3.881, 5.820, 8.365, 11.63, 15.75, 20.96, 27.65, 36.71, 50.89, 99.75]),
        "c_p_liquid": np.array([31.58, 30.58, 29.87, 29.37, 29.05, 28.91, 28.94, 29.12, 29.45, 29.94, 30.58, 31.37, 32.32, 33.44, 34.85, 36.67, 39.14, 42.70, 48.30, 58.60, 84.75, 482.9]),
        "c_v_liquid": np.array([28.28, 26.46, 24.94, 23.66, 22.59, 21.71, 21.03, 20.54, 20.22, 20.07, 20.10, 20.30, 20.69, 21.27, 21.99, 22.88, 23.99, 25.45, 27.51, 30.91, 38.69, 135.8]),
        "c_p_vapor": np.array([23.52, 30.27, 40.12, 49.66, 56.38, 59.70, 60.32, 59.37, 57.81, 56.26, 54.97, 53.92, 52.90, 51.88, 51.98, 53.55, 57.05, 63.60, 76.02, 103.4, 194.9, 4123.0]),
        "c_v_vapor": np.array([15.16, 21.14, 29.45, 36.93, 41.57, 43.19, 42.66, 41.01, 38.93, 36.83, 34.83, 32.93, 31.07, 29.28, 28.05, 27.42, 27.39, 28.07, 29.85, 34.03, 47.20, 366.8]),
        "vapor_pressure": np.array([1.358e-10, 7.635e-8, 5.047e-6, 9.869e-5, 9.043e-4, 5.010e-3, 1.955e-2, 5.917e-2, 0.1482, 0.3209, 0.6203, 1.096, 1.798, 2.779, 4.087, 5.766, 7.851, 10.37, 13.36, 16.81, 20.76, 25.19]) * 1e6,
        "viscosity": np.array([0.614, 0.416, 0.320, 0.265, 0.229, 0.204, 0.185, 0.171, 0.160, 0.150, 0.142, 0.136, 0.130, 0.125, 0.120, 0.116, 0.112, 0.108, 0.105, 0.102, 0.0992, 0.0965]) * 1e-3,
        "heat_of_vaporization": np.array([103.3, 102.1, 100.6, 98.78, 96.77, 94.63, 92.47, 90.32, 88.17, 85.98, 83.68, 81.21, 78.53, 75.64, 72.49, 69.03, 65.15, 60.72, 55.48, 48.93, 39.69, 18.03]),
        "enthalpy_liquid": np.array([5.669, 8.774, 11.79, 14.75, 17.67, 20.57, 23.46, 26.36, 29.29, 32.26, 35.30, 38.40, 41.59, 44.89, 48.32, 51.90, 55.69, 59.76, 64.23, 69.36, 75.83, 88.52]),
        "enthalpy_vapor": np.array([109.0, 110.8, 112.4, 113.5, 114.4, 115.2, 115.9, 116.7, 117.5, 118.2, 119.0, 119.6, 120.1, 120.5, 120.8, 120.9, 120.8, 120.5, 119.7, 118.3, 115.5, 106.6]),
        "thermal_conductivity": np.array([86.8, 81.8, 76.4, 71.5, 66.8, 62.5, 58.3, 54.3, 50.3, 46.5, 42.2, 39.3, 35.8, 32.3, 28.9, 25.5, 22.2, 18.9, 15.6, 12.3, 9.5, 5.8])
    }

def fit_and_generate_symbolic_function(T_data, Y_data, model_func):
    params, _ = curve_fit(model_func, T_data, Y_data)
    def symbolic(T):
        return model_func(T, *params)
    print(f"Fitted parameters: {params}")
    return symbolic

def get_symbolic_sodium_properties(tabulated_data: dict) -> dict:
    """
    Return a dictionary of symbolic functions for sodium properties.
    
    Returns:
        Dictionary of symbolic functions for sodium properties:
        density - Density (kg/m³)
        specific_heat - Specific heat (J/kg-K)
        vapor_pressure - Vapor pressure (Pa)
        viscosity - Dynamic viscosity (Pa-s)
        heat_of_vaporization - Heat of vaporization (J/kg)
        enthalpy - Enthalpy (J/kg)
        thermal_conductivity - Thermal conductivity (W/m-K)
    """
    T_data = np.arange(400, 2501, 100)

    # Density
    T_data = np.arange(400, 2501, 100) 
    rho_data_liquid = tabulated_data["density_liquid"]
    rho_liquid_model = lambda T, a, b: a * T + b
    rho_fit_liquid = fit_and_generate_symbolic_function(T_data, rho_data_liquid, rho_liquid_model)
    
    # Specific Heat
    T_data = np.arange(400, 2501, 100)
    c_p_data_liquid = tabulated_data["c_p_liquid"]
    c_v_data_liquid = tabulated_data["c_v_liquid"]
    c_p_data_vapor = tabulated_data["c_p_vapor"]
    c_v_data_vapor = tabulated_data["c_v_vapor"]
    c_model = lambda T, a, b, c, d, e, f, g: a + b * T + c * T**2 + d * T**3 + e * T**4 + f * T**5 + g * T**6
    c_p_fit_liquid = fit_and_generate_symbolic_function(T_data, c_p_data_liquid, c_model)
    c_v_fit_liquid = fit_and_generate_symbolic_function(T_data, c_v_data_liquid, c_model)
    c_p_fit_vapor = fit_and_generate_symbolic_function(T_data, c_p_data_vapor, c_model)
    c_v_fit_vapor = fit_and_generate_symbolic_function(T_data, c_v_data_vapor, c_model)

    # Vapor Pressure
    T_data = np.arange(400, 2501, 100)
    P_data = tabulated_data["vapor_pressure"]
    P_data = P_data * 1e6  # Convert from MPa to Pa
    P_model = lambda T, a, b, c: npx.exp(a + b / T + c * npx.log(T))*1e-6
    P_fit = fit_and_generate_symbolic_function(T_data, P_data, P_model)

    # Viscosity
    mu_data = tabulated_data["viscosity"]
    mu_model = lambda T, a, b, c: a * T**(-b) + c
    mu_fit = fit_and_generate_symbolic_function(T_data, mu_data, mu_model)

    # Heat of Vaporization
    T_data = np.arange(400, 2501, 100)
    h_lv_data = tabulated_data["heat_of_vaporization"]
    h_lv_model = lambda T, a, b, c: a * T**b + c
    h_lv_fit = fit_and_generate_symbolic_function(T_data, h_lv_data, h_lv_model)

    # Enthalpy
    T_data = np.arange(400, 2501, 100)
    h_v_data = tabulated_data["enthalpy_vapor"]
    h_l_data = tabulated_data["enthalpy_liquid"]
    h_model = lambda T, a, b, c: a * T**2 + b * T + c
    h_fit_liquid = fit_and_generate_symbolic_function(T_data, h_l_data, h_model)
    h_fit_vapor = fit_and_generate_symbolic_function(T_data, h_v_data, h_model)

    # Thermal Conductivity
    T_data = np.arange(400, 2501, 100)
    k_data = tabulated_data["thermal_conductivity"]
    k_model = lambda T, a, b, c: a * T**2 + b * T + c
    k_fit = fit_and_generate_symbolic_function(T_data, k_data, k_model)

    # Special Cases: STILL NEEDS TO BE IMPLEMENTED CORRECTLY!!! TODO! 
    # For the density of sodium vapor, we need to fit a different model
    def dP_satdT(T):
        return npx.exp(19777.0 * T / 100e6 - 13113 / T + 2354 / 125) / (T ** (1373 / 1250))
    
    def rho_vapor_symbolic(T):
        h_vap = h_lv_fit(T)
        dPdT = dP_satdT(T)
        rho_liquid = rho_fit_liquid(T)
        return (h_vap / (T * dPdT) + 1 / rho_liquid) ** -1 
    
    rho_fit_vapor = rho_vapor_symbolic

    return {
        "density_liquid": rho_fit_liquid,
        "density_vapor": rho_fit_vapor,
        "c_p_liquid": c_p_fit_liquid,
        "c_p_vapor": c_p_fit_vapor,
        "c_v_liquid": c_v_fit_liquid,
        "c_v_vapor": c_v_fit_vapor,
        "vapor_pressure": P_fit,
        "viscosity": mu_fit,
        "heat_of_vaporization": h_lv_fit,
        "enthalpy_liquid": h_fit_liquid,
        "enthalpy_vapor": h_fit_vapor,
        "thermal_conductivity": k_fit
    }

tabulated_data = get_tabulated_data()
sodium_properties = get_symbolic_sodium_properties(tabulated_data)
    
def compare_tabulated_and_interpolated_data():
    """
    Compare tabulated sodium property data with the fitted symbolic functions by creating plots
    for each property.
    """
    import matplotlib.pyplot as plt
    
    # Temperature data points
    T_data = np.arange(400, 2501, 100)
    # Temperature range for smoother fitted curve
    T_fit = np.linspace(400, 2500, 1000)
    
    # Property names and labels for plotting
    properties = {
        "density_liquid": ("Density of Liquid Sodium", "Density (kg/m³)"),
        "density_vapor": ("Density of Sodium Vapor", "Density (kg/m³)"),
        "c_p_liquid": ("Specific Heat (Cp) of Liquid Sodium", "Specific Heat (J/kg-K)"),
        "c_p_vapor": ("Specific Heat (Cp) of Sodium Vapor", "Specific Heat (J/kg-K)"),
        "c_v_liquid": ("Specific Heat (Cv) of Liquid Sodium", "Specific Heat (J/kg-K)"),
        "c_v_vapor": ("Specific Heat (Cv) of Sodium Vapor", "Specific Heat (J/kg-K)"),
        "vapor_pressure": ("Vapor Pressure of Sodium", "Pressure (Pa)"),
        "viscosity": ("Dynamic Viscosity of Sodium", "Viscosity (Pa-s)"),
        "heat_of_vaporization": ("Heat of Vaporization of Sodium", "Heat of Vaporization (J/kg)"),
        "enthalpy_liquid": ("Enthalpy of Liquid Sodium", "Enthalpy (J/kg)"),
        "enthalpy_vapor": ("Enthalpy of Sodium Vapor", "Enthalpy (J/kg)"),
        "thermal_conductivity": ("Thermal Conductivity of Sodium", "Thermal Conductivity (W/m-K)")
    }
    
    # Create plots for each property
    for prop, (title, ylabel) in properties.items():
        # Get tabulated data
        tabulated_values = tabulated_data[prop]
        
        # Calculate fitted values
        fitted_values = np.array([sodium_properties[prop](T) for T in T_fit])
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(T_data, tabulated_values, 'o', label='Tabulated Data', markersize=5)
        plt.plot(T_fit, fitted_values, '-', label='Fitted Curve', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Use log scale for properties with large variations
        if prop in ["density_vapor", "vapor_pressure"]:
            plt.yscale('log')
            
        plt.tight_layout()
        plt.savefig(f"{prop}_comparison.png")
        plt.show()

# To create all the plots, call this function:
compare_tabulated_and_interpolated_data()

# ----------------------------------------
# Material Properties of Stainless Steel
# ----------------------------------------

def steel_density(T):
    """
    Density of stainless steel as a function of temperature. **USE THIS!** USAGE REGION: wall.
    """
    return (7.9841 - 2.6506e-4 * T - 1.1580e-7 * T**2) * 1e3 

def steel_thermal_conductivity(T):
    """
    Thermal conductivity of stainless steel as a function of temperature. **USE THIS!** USAGE REGION: wall.
    """
    return (8.116e-2 + 1.618e-4 * T) * 100

def steel_specific_heat() -> interp1d:
    """
    Return interpolation function for stainless steel specific heat as a function of temperature. **USE THIS!** USAGE REGION: wall. 
    """
    
    T_data = np.arange(300, 1701, 100)

    c_p_data = np.array([510.0296, 523.4184, 536.8072, 550.1960, 564.0032, 577.3920, 590.7808, 604.1696, 617.5584, 631.3656, 644.7544, 658.1432, 671.5320, 685.3392, 698.7280])

    specific_heat_interp = interp1d(T_data, c_p_data, kind='cubic', fill_value='extrapolate')
    return specific_heat_interp

def get_steel_properties() -> dict:
    """
    Return a dictionary of interpolation functions for stainless steel properties.
    
    Returns:
        Dictionary of interpolation functions for stainless steel properties:
        density - Density (kg/m³)
        specific_heat - Specific heat (J/kg-K)
        thermal_conductivity - Thermal conductivity (W/m-K)
    """
    return {
        "density": steel_density,
        "specific_heat": steel_specific_heat(),
        "thermal_conductivity": steel_thermal_conductivity
    }

# ----------------------------------------
# Effective Values for Vapor Core and Wick
# ----------------------------------------

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
    
def get_c_p_Na_i(T: CellVariable, sodium_parameters:dict, parameters: dict, c_p_l: float, c_p_s: float) -> float:
    """
    Effective specific heat for sodium inside the wick region. Melting point is considered.
    """
    delta_T = parameters['delta_T_sodium']
    T_m = parameters['T_melting_sodium']
    h_vap = sodium_parameters['heat_of_vaporization'](T)

    if T < T_m - delta_T:
        return c_p_s
    elif T_m - delta_T <= T <= T_m + delta_T:
        return (c_p_s + c_p_l) / 2 + h_vap / (2 * delta_T)
    else:
        return c_p_l

def get_rho_Na_i(T: CellVariable, sodium_parameters: dict, parameters: dict, rho_l: float, rho_s: float) -> float:
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
    Effective thermal conductivity for the vapor core region. **USE THIS!** USAGE REGION: vapor core.
    
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
        res = -((R_v**2 * P) / (4 * mu_v) + (2 * R_v) / 3 * np.sqrt((8 * R * T) / (np.pi * M_g))) * (h_l * h_vap * M_g**2 * P) / (R**2 * T**3)
    elif region == 'adiabatic':
        res = -((R_v**2 * P) / (8 * mu_v) + (2 * R_v) / 3 * np.sqrt((8 * R * T) / (np.pi * M_g))) * (h_v * h_vap * M_g**2 * P) / (R**2 * T**3)
    else:
        raise ValueError("Invalid region specified. Use 'evap_cond' or 'adiabatic'.")
    
    return res

def k_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict) -> float:
    """
    Effective thermal conductivity for the wick region. **USE THIS!** USAGE REGION: wick.
    """
    k_s = parameters['k_solid_sodium']
    k_l = sodium_properties['thermal_conductivity'](T)

    k_steel = steel_properties['thermal_conductivity'](T)
    epsilon = parameters['porosity_wick']

    k_Na_i = get_k_Na_i(T, parameters, k_l, k_s)

    return k_Na_i * ((k_Na_i + k_steel) - (1 - epsilon) * (k_Na_i - k_steel)) / ((k_Na_i + k_steel) + (1 - epsilon) * (k_Na_i - k_steel))

def c_p_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict) -> float:
    """
    Effective specific heat for the wick region. **USE THIS!** USAGE REGION: wick.
    """
    c_p_l = sodium_properties['c_p_liquid'](T)
    c_p_s = parameters['c_p_solid_sodium']
    c_p_Na_i = get_c_p_Na_i(T, parameters, c_p_l, c_p_s)
    epsilon = parameters['porosity_wick']

    return epsilon * c_p_Na_i + (1 - epsilon) * steel_properties['specific_heat'](T)

def rho_eff_wick(T: CellVariable, sodium_properties: dict, steel_properties: dict, parameters: dict) -> float:
    """
    Effective density for the wick region. **USE THIS!** USAGE REGION: wick.
    """
    rho_l = sodium_properties['density_liquid'](T)
    rho_s = parameters['rho_solid_sodium']
    rho_Na_i = rho_Na_i(T, parameters, rho_l, rho_s)
    epsilon = parameters['porosity_wick']

    return epsilon * rho_Na_i + (1 - epsilon) * steel_properties['density'](T)

def get_wick_properties() -> dict:
    """
    Return a dictionary of effective properties for the wick region.
    
    Returns:
        Dictionary of effective properties for the wick region:
        k_eff - Effective thermal conductivity (W/m-K)
        c_p_eff - Effective specific heat (J/kg-K)
        rho_eff - Effective density (kg/m³)
    """
    return {
        "thermal_conductivity": k_eff_wick,
        "specific_heat": c_p_eff_wick,
        "density    ": rho_eff_wick
    }

# ----------------------------------------
# Sonic Limit
# ----------------------------------------

def Q_sonic(T: CellVariable, sodium_properties: dict, dimensions: dict, constants: dict) -> float:
    """
    Calculate the sonic heat transfer rate in the vapor core region.
    """
    rho_v_0 = sodium_properties['density_vapor'](T[0]) # density at the evaporator end cap
    T_v_0 = T[0]
    A_vc = np.pi * dimensions['R_vc']**2
    h_vap = sodium_properties['heat_of_vaporization'](T)
    gamma = sodium_properties['c_p_vapor'](T) / sodium_properties['c_v_vapor'](T)
    R_g = constants['R']

    return (rho_v_0 * A_vc * h_vap * np.sqrt(gamma * R_g * T_v_0)) / (np.sqrt(2 * (gamma + 1)))

# Testing
if False:
    sodium_properties = get_sodium_properties()
    T = 420
    print(f"Sodium density at {T} K: {sodium_properties['density'](T)} kg/m³")
    print(f"Sodium specific heat at {T} K: {sodium_properties['specific_heat'](T)} J/kg-K")
    print(f"Sodium vapor pressure at {T} K: {sodium_properties['vapor_pressure'](T)} Pa")
    print(f"Sodium dynamic viscosity at {T} K: {sodium_properties['viscosity'](T)} Pa-s")
    print(f"Sodium heat of vaporization at {T} K: {sodium_properties['heat_of_vaporization'](T)} J/kg")
    print(f"Sodium enthalpy at {T} K: {sodium_properties['enthalpy'](T)} J/kg")
    print(f"Sodium thermal conductivity at {T} K: {sodium_properties['thermal_conductivity'](T)} W/m-K")