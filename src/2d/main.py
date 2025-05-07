#!/usr/bin/env python
"""
main.py - Driver script for the 2D heat pipe conduction model using FiPy.
Assumes that functions for meshing, boundary conditions, k_eff computation, etc.,
are provided in separate modules.
"""

import matplotlib.pyplot as plt
from fipy import CellVariable, TransientTerm, DiffusionTerm, Viewer, FaceVariable
from fipy.tools import numerix as npx
from tqdm import tqdm
from mesh import generate_mesh_2d, generate_composite_mesh
from params import get_all_params, get_param_group
from utils import preview_mesh, preview_face_mask, save_animation, init_tripcolor_viewer, preview_cell_mask
from material_properties import get_sodium_properties, get_steel_properties, get_vc_properties, get_wick_properties

# ----------------------------------------
# Load parameters from configuration file
# ----------------------------------------

all_params = get_all_params()
mesh_params = get_param_group('mesh')
dimensions = get_param_group('dimensions')
parameters = get_param_group('parameters')
constants = get_param_group('constants')

L_total = all_params['L_e'] + all_params['L_a'] + all_params['L_c']
R_total = all_params['R_wick'] + all_params['R_wall'] + all_params['R_vc']

# ----------------------------------------
# Load region-specific material properties
# ----------------------------------------

sodium_properties = get_sodium_properties()
steel_properties = get_steel_properties()
vc_properties = get_vc_properties()
wick_properties = get_wick_properties()

# ----------------------------------------
# Generate the 2D mesh
# ----------------------------------------

mesh, cell_types = generate_composite_mesh(mesh_params, dimensions) 
# Base radial types: vapor core=0, wick=10, wall=20
# Add axial types: evaporator/condenser=+0, adiabatic=+1

x_cell, y_cell = mesh.cellCenters

# preview_mesh(mesh, title="2D Mesh Preview")

# ----------------------------------------
# Define the primary variable (temperature)
# ----------------------------------------

T = CellVariable(name="Temperature", mesh=mesh, value=all_params["T_amb"])

# ----------------------------------------
# Define material properties (Tempertaure-dependent!)
# ----------------------------------------

# Wall properties
k_wall = steel_properties['thermal_conductivity'](T) # CellTypes 20 and 21
c_p_wall = steel_properties['specific_heat'](T) # CellTypes 20 and 21
rho_wall_evap_adia = steel_properties['density_evap_adia'](T) # CellTypes 20 and 21 and only in the evaporator and adiabatic regions
rho_wall_cond = steel_properties['density_cond'](T) # CellTypes 20 and 21 and only in the condenser region

# Wick properties
k_wick = wick_properties['thermal_conductivity'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
c_p_wick = wick_properties['specific_heat'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
rho_wick = wick_properties['density'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11


# Vapor core properties
k_vc_evap_cond = vc_properties['thermal_conductivity'](T, mesh, 'evap_cond', sodium_properties, dimensions, parameters, constants) # CellType 0
k_vc_adiabatic = vc_properties['thermal_conductivity'](T, mesh, 'adiabatic', sodium_properties, dimensions, parameters, constants) # CellType 1
c_p_vc = vc_properties['specific_heat'](T) # CellTypes 0 and 1
rho_vc = vc_properties['density'](T) # CellTypes 0 and 1

# ----------------------------------------
# Define the spatially varying D coefficient 
# ----------------------------------------

# Convenience masks
vc_evap_cond = (cell_types == 0)
vc_adiabatic = (cell_types == 1)
wick = (cell_types == 10) | (cell_types == 11)
wall = (cell_types == 20) | (cell_types == 21)
wall_cond = wall & (x_cell > dimensions['L_e'] + dimensions['L_a'])
wall_evap_adia = wall & (x_cell < dimensions['L_e'] + dimensions['L_a']) # Wall in evaporator and adiabatic regions

preview_cell_mask(mesh, wall_cond, title="Wall Condenser Region")
# preview_cell_mask(mesh, wall_evap_adia, title="Wall Evaporator and Adiabatic Region")

# Calculate D = k / (rho * c_p)
D_expr = 0 * T
epsilon = 1e-12

# Add symbolic contributions by region
D_expr = D_expr + ((cell_types == 0) * (k_vc_evap_cond / (rho_vc * c_p_vc + epsilon)))
D_expr = D_expr + ((cell_types == 1) * (k_vc_adiabatic / (rho_vc * c_p_vc + epsilon)))
D_expr = D_expr + ((wick) * (k_wick / (rho_wick * c_p_wick + epsilon)))
D_expr = D_expr + ((wall_cond) * (k_wall / (rho_wall_cond * c_p_wall + epsilon)))
D_expr = D_expr + ((wall_evap_adia) * (k_wall / (rho_wall_evap_adia * c_p_wall + epsilon)))

# ----------------------------------------
# Define the PDE
# ----------------------------------------

eq = TransientTerm(var=T) == DiffusionTerm(coeff=D_expr, var=T)

# ----------------------------------------
# Apply boundary conditions
# ----------------------------------------

X, Y = mesh.faceCenters
faces_evaporator = (mesh.facesTop & ((X < dimensions['L_input_right']) & (X > dimensions['L_input_left'])))
faces_condenser = (mesh.facesTop & ((X > dimensions['L_e'] + dimensions['L_a']) & (X < L_total)))
faces_end_cap_vc = mesh.facesLeft & (Y <= dimensions['R_vc'])

x_cell, y_cell = mesh.cellCenters
cells_evaporator = (cell_types == 0) & (x_cell < dimensions['L_input_right']) & (x_cell > dimensions['L_input_left'])
cells_condenser = (cell_types == 0) & ((x_cell > dimensions['L_e'] + dimensions['L_a']) & (x_cell < L_total))

# preview_face_mask(mesh, faces_evaporator, title="Evaporator Face Mask")
# preview_face_mask(mesh, faces_condenser, title="Condenser Face Mask")

# Define face-normal unit vectors
n = mesh.faceNormals

# Apply the flux boundary condition at the evaporator faces
# TODO: DISCUSS TAKING THE MEAN
T_evap = T.value[cells_evaporator].mean()  # Get temperature values at the evaporator faces
q_flux = parameters['Q_input_flux'] / steel_properties['thermal_conductivity'](T_evap)  # Calculate the heat flux (W/m^2)
T.faceGrad.constrain(q_flux * n, where=faces_evaporator)

T_rad = T.value[cells_condenser].mean()  # Get temperature values at the condenser faces
q_rad = constants['sigma'] * parameters['emissivity'] * (T_rad**4 - parameters['T_amb']**4) # Radiative heat flux (W/m^2)
T.faceGrad.constrain(-q_rad / steel_properties['thermal_conductivity'](T_rad) * n, where=faces_condenser)

# ----------------------------------------
# Time-stepping loop
# ----------------------------------------

# if __name__ == "__main__":
#     viewer = Viewer(vars=T, title="Temperature Distribution")

time_step = all_params['dt']
steps = int(all_params['t_end'] / time_step)

temperature_evolution = []  # List to store temperature values over time
time_values = []  # List to store time values
measure_times = [500, 10000, 14000]  # Time points to measure temperature (t / dt)

# Mask for vapor core cells (cell_types == 0)
vapor_core_mask = (cell_types == 0)

# Prepare to store temperature profiles at measure_times
profiles = []
profile_times = []

# from fipy import LinearLUSolver
# solver = LinearLUSolver(tolerance=1e-8, iterations=1000)

# Run the simulation and store T at measure_times
T.setValue(all_params["T_amb"])  # Reset temperature
for step in tqdm(range(steps)):
    eq.solve(var=T, dt=time_step)
    # viewer.plot()
    if step in measure_times:
        profiles.append(T.value[wall].copy())
        profile_times.append(step * time_step)


# Get x-coordinates for vapor core cells
x_vapor = x_cell[vapor_core_mask]

# Sort by x for plotting
sort_idx = x_vapor.argsort()
x_vapor_sorted = x_vapor[sort_idx]

plt.figure(figsize=(10, 6))
for i, (T_profile, t) in enumerate(zip(profiles, profile_times)):
    plt.plot(x_vapor_sorted, T_profile[sort_idx], label=f"t = {t:.2f} s")
plt.xlabel("x [m]")
plt.ylabel("Temperature [K]")
plt.title("Axial Temperature Profile in Vapor Core at Different Times")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/vapor_core_axial_profiles.png", dpi=300)
plt.show()
