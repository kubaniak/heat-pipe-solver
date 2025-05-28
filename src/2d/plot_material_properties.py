import numpy as np
from material_properties import get_steel_properties, get_sodium_properties, get_wick_properties, get_vc_properties
from matplotlib import pyplot as plt
from params import get_all_params
from mesh import generate_composite_mesh
import os

# Create a directory for material property plots if it doesn't exist
output_dir = "plots/material_properties"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Na_props = get_sodium_properties()
steel_props = get_steel_properties()
params = get_all_params()
# The mesh, dimensions, and constants are needed for some property functions
# For plotting vs temperature, we don't need a complex mesh,
# but some functions might expect certain mesh attributes or cell/face variables.
# We'll use a simplified approach or mock objects if necessary,
# focusing on the temperature dependency.
# However, `generate_composite_mesh` returns mesh and cell_types,
# and `get_all_params` returns a dictionary.
# `main_clean.py` also defines `dimensions` and `constants` from `params`.
dimensions = params # Assuming dimensions are part of params or can be derived
constants = params # Assuming constants are part of params or can be derived

# For functions requiring mesh, we pass the generated mesh.
# For T.faceValue or T, we will pass a numpy array of temperatures.
# Some properties are defined on faces (like k) and some on cells (like cp, rho).
# For plotting purposes, we'll evaluate them with a temperature array.

mesh, _ = generate_composite_mesh(params, dimensions)


vc_properties = get_vc_properties()
wick_properties = get_wick_properties()

T_range = np.linspace(300, 1500, 500) # Temperature range in Kelvin

# --- Plotting VC Properties ---

# VC Thermal Conductivity (Evap/Cond)
plt.figure()
y_values = vc_properties['thermal_conductivity'](T_range, mesh, 'evap_cond', Na_props, dimensions, params, constants)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal Conductivity (W/m-K)")
plt.title("VC Thermal Conductivity (Evap/Cond) vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "vc_thermal_conductivity_evap_cond.png"))
plt.close()

# VC Thermal Conductivity (Adiabatic)
plt.figure()
y_values = vc_properties['thermal_conductivity'](T_range, mesh, 'adiabatic', Na_props, dimensions, params, constants)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal Conductivity (W/m-K)")
plt.title("VC Thermal Conductivity (Adiabatic) vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "vc_thermal_conductivity_adiabatic.png"))
plt.close()

# VC Specific Heat
plt.figure()
y_values = vc_properties['specific_heat'](T_range)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Specific Heat (J/kg-K)")
plt.title("VC Specific Heat vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "vc_specific_heat.png"))
plt.close()

# VC Density
plt.figure()
y_values = vc_properties['density'](T_range)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Density (kg/m^3)")
plt.title("VC Density vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "vc_density.png"))
plt.close()

# --- Plotting Wick Properties ---

# Wick Thermal Conductivity
plt.figure()
y_values = wick_properties['thermal_conductivity'](T_range, Na_props, steel_props, params)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal Conductivity (W/m-K)")
plt.title("Wick Thermal Conductivity vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "wick_thermal_conductivity.png"))
plt.close()

# Wick Specific Heat
plt.figure()
y_values = wick_properties['specific_heat'](T_range, Na_props, steel_props, params)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Specific Heat (J/kg-K)")
plt.title("Wick Specific Heat vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "wick_specific_heat.png"))
plt.close()

# Wick Density
plt.figure()
y_values = wick_properties['density'](T_range, Na_props, steel_props, params)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Density (kg/m^3)")
plt.title("Wick Density vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "wick_density.png"))
plt.close()

# --- Plotting Steel Properties ---

# Steel Thermal Conductivity
plt.figure()
y_values = steel_props['thermal_conductivity'](T_range)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal Conductivity (W/m-K)")
plt.title("Steel Thermal Conductivity vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "steel_thermal_conductivity.png"))
plt.close()

# Steel Specific Heat
plt.figure()
y_values = steel_props['specific_heat'](T_range)
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Specific Heat (J/kg-K)")
plt.title("Steel Specific Heat vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "steel_specific_heat.png"))
plt.close()

# Steel Density (Evap/Adia) - Assuming 'density_evap_adia' is the correct key from main_clean.py context
# and it takes only Temperature as argument like other steel properties.
# If 'density_evap_adia' is different from 'density' in steel_props, it needs to be handled.
# From material_properties.py, steel_props has 'density_evap_adia' and 'density'.
# Let's assume 'density_evap_adia' is the one for this case.
plt.figure()
y_values = steel_props['density_evap_adia'](T_range) # or steel_props['density'](T_range) if that's intended
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Density (kg/m^3)")
plt.title("Steel Density (Evap/Adia) vs. Temperature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "steel_density_evap_adia.png"))
plt.close()

# Steel Density (Cond) - Assuming 'density_cond' is available or same as 'density'
# main_clean.py shows steel_properties['density_cond'](T)
# material_properties.py shows get_steel_properties() returns 'density' and 'density_evap_adia'
# It seems 'density_cond' might be the same as 'density' or 'density_evap_adia' or a specific one.
# For now, let's use 'density' as a placeholder if 'density_cond' is not directly in steel_props.
# Based on src/2d/material_properties.py, get_steel_properties returns:
# 'density_evap_adia': rho_eff_steel,
# 'density': rho_steel,
# 'thermal_conductivity': k_steel,
# 'specific_heat': c_p_steel,
# So, 'density_cond' seems to map to 'density' (rho_steel)
plt.figure()
if 'density_cond' in steel_props: # Check if a specific 'density_cond' exists
    y_values = steel_props['density_cond'](T_range)
    title = "Steel Density (Cond) vs. Temperature"
else: # Fallback to the general 'density' if 'density_cond' is not a separate key
    y_values = steel_props['density'](T_range)
    title = "Steel Density (Cond - using general steel_props['density']) vs. Temperature"
plt.plot(T_range, y_values)
plt.xlabel("Temperature (K)")
plt.ylabel("Density (kg/m^3)")
plt.title(title)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "steel_density_cond.png"))
plt.close()

print(f"Plots saved in {output_dir}")

