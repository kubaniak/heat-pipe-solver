"""
model_fit_visualizer.py

This module contains functions to fit model functions to tabulated data and visualize the results.
It can be used to find coefficients for various material property models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable, Tuple, Dict, Any, List

def fit_model(T_data: np.ndarray, y_data: np.ndarray, model_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a model function to the provided data.
    
    Args:
        T_data: Temperature data points
        y_data: Corresponding y-values (property data points)
        model_func: The model function to fit
        
    Returns:
        Tuple containing the fitted parameters and the covariance matrix
    """
    params, covariance = curve_fit(model_func, T_data, y_data)
    return params, covariance

def generate_fitted_values(T_range: np.ndarray, model_func: Callable, params: np.ndarray) -> np.ndarray:
    """
    Generate fitted values using the model function and fitted parameters.
    
    Args:
        T_range: Temperature range for evaluation
        model_func: The model function
        params: Fitted parameters
        
    Returns:
        Array of fitted values
    """
    return np.array([model_func(T, *params) for T in T_range])

def visualize_fit(T_data: np.ndarray, y_data: np.ndarray, T_range: np.ndarray, fitted_values: np.ndarray, 
                 title: str, ylabel: str, use_log_scale: bool = False) -> None:
    """
    Visualize the original data points and the fitted curve.
    
    Args:
        T_data: Temperature data points
        y_data: Corresponding y-values (property data points)
        T_range: Temperature range for the fitted curve
        fitted_values: Fitted values
        title: Plot title
        ylabel: Y-axis label
        use_log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    plt.plot(T_data, y_data, 'o', label='Original Data', markersize=5)
    plt.plot(T_range, fitted_values, '-', label='Fitted Curve', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if use_log_scale:
        plt.yscale('log')
        
    plt.tight_layout()

def analyze_property(T_data: np.ndarray, y_data: np.ndarray, model_func: Callable, 
                    property_name: str, unit: str, use_log_scale: bool = False) -> np.ndarray:
    """
    Analyze a property by fitting a model and visualizing the results.
    
    Args:
        T_data: Temperature data points
        y_data: Corresponding property values
        model_func: The model function to fit
        property_name: Name of the property
        unit: Unit of the property
        use_log_scale: Whether to use log scale for y-axis
        
    Returns:
        Fitted parameters
    """
    # Fit the model
    params, _ = fit_model(T_data, y_data, model_func)
    
    # Generate a smooth temperature range for plotting
    T_smooth = np.linspace(np.min(T_data), np.max(T_data), 1000)
    
    # Generate fitted values
    fitted_values = generate_fitted_values(T_smooth, model_func, params)
    
    # Visualize the fit
    title = f"{property_name} vs. Temperature"
    ylabel = f"{property_name} ({unit})"
    visualize_fit(T_data, y_data, T_smooth, fitted_values, title, ylabel, use_log_scale)
    
    # Save the plot
    # filename = f"{property_name.lower().replace(' ', '_')}_fit.png"
    # plt.savefig(filename)
    plt.show()
    
    # Print the fitted parameters
    print(f"\nFitted parameters for {property_name}:")
    for i, param in enumerate(params):
        print(f"Parameter {i+1}: {param}")
        
    # Calculate and print fit quality metrics
    y_fitted = generate_fitted_values(T_data, model_func, params)
    residuals = y_data - y_fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    
    print(f"R² value: {r_squared:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    return params

def get_model_function(model_type: str) -> Callable:
    """
    Get a predefined model function by type.
    
    Args:
        model_type: Type of model function
        
    Returns:
        Model function
    """
    model_functions = {
        'linear': lambda T, a, b: a * T + b,
        'quadratic': lambda T, a, b, c: a * T**2 + b * T + c,
        'cubic': lambda T, a, b, c, d: a * T**3 + b * T**2 + c * T + d,
        'polynomial_6': lambda T, a, b, c, d, e, f, g: a + b * T + c * T**2 + d * T**3 + e * T**4 + f * T**5 + g * T**6,
        'power_law': lambda T, a, b, c: a * T**b + c,
        'exponential': lambda T, a, b, c: np.exp(a + b / T + c * np.log(T)),
        'inverse_power': lambda T, a, b, c: a * T**(-b) + c,
        'codata': lambda T, a, b, c, d: a + b * T + c * T**2 + d / T**2,
    }
    
    if model_type in model_functions:
        return model_functions[model_type]
    else:
        raise ValueError(f"Model type '{model_type}' not recognized. Available types: {list(model_functions.keys())}")

def main():
    """
    Example usage of the module.
    """
    # Example: Fit a model to sodium vapor density data
    # Define the data
    T_data = np.arange(400, 2401, 100)
    
    # This is just example data - replace with your actual data
    c_p_data = np.array([
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
        8030
    ])
    
    # Get a model function
    model_func = get_model_function('codata') 
    
    # Analyze the property
    params = analyze_property(
        T_data, 
        c_p_data, 
        model_func, 
        "Specific Heat Capacity of Vapor Sodium",
        "J/(kg*K)",
        use_log_scale=False
    )
    # a, b = params
    print(f"\nFitted model equation:")
    print(f"c_p(T) = {params[0]:.4f} + {params[1]:.4f} * T + {params[2]:.4f} * T**2 + {params[3]:.4f} / T**2")

if __name__ == "__main__":
    main()