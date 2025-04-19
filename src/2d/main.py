#!/usr/bin/env python
"""
main.py - Driver script for the 2D heat pipe conduction model using FiPy.
Assumes that functions for meshing, boundary conditions, k_eff computation, etc.,
are provided in separate modules.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from fipy import CellVariable, TransientTerm, DiffusionTerm, Viewer, FaceVariable
from fipy.tools import numerix as npx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from mesh import generate_mesh_2d, generate_composite_mesh
from params import get_all_params, get_param_group
from utils import preview_mesh, preview_face_mask, save_animation, init_tripcolor_viewer
from material_properties import get_sodium_properties, get_steel_properties, get_vc_properties, get_wick_properties

from fipy import __version__ as fipy_version
logger.info(f"Using FiPy version: {fipy_version}")

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

logger.info("Parameters Loaded")

# ----------------------------------------
# Load region-specific material properties
# ----------------------------------------

sodium_properties = get_sodium_properties()
steel_properties = get_steel_properties()
vc_properties = get_vc_properties()
wick_properties = get_wick_properties()

logger.info("Material Properties Loaded")

# ----------------------------------------
# Generate the 2D mesh
# ----------------------------------------

mesh, cell_types = generate_composite_mesh(mesh_params, dimensions) 
# Base radial types: vapor core=0, wick=10, wall=20
# Add axial types: evaporator/condenser=+0, adiabatic=+1

logger.info("Mesh Generated")
logger.info(f"Mesh shape: {mesh.shape}")

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
rho_wall = steel_properties['density'](T) # CellTypes 20 and 21

# Wick properties
k_wick = wick_properties['thermal_conductivity'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
c_p_wick = wick_properties['specific_heat'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
rho_wick = wick_properties['density'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11

# Vapor core properties
k_vc_evap_cond = vc_properties['thermal_conductivity'](T, 'evap_cond', sodium_properties, dimensions, parameters, constants) # CellType 0
k_vc_adiabatic = vc_properties['thermal_conductivity'](T, 'adiabatic', sodium_properties, dimensions, parameters, constants) # CellType 1
c_p_vc = vc_properties['specific_heat'](T) # CellTypes 0 and 1
rho_vc = vc_properties['density'](T) # CellTypes 0 and 1

# ----------------------------------------
# Define the spatially varying D coefficient 
# ----------------------------------------

# Predefine empty arrays
k = npx.zeros(mesh.numberOfCells)
rho = npx.zeros(mesh.numberOfCells)
c_p = npx.zeros(mesh.numberOfCells)

# Convenience masks
vc_evap_cond = (cell_types == 0)
vc_adiabatic = (cell_types == 1)
wick = (cell_types == 10) | (cell_types == 11)
wall = (cell_types == 20) | (cell_types == 21)

# Assign values for each region
k[vc_evap_cond] = k_vc_evap_cond[vc_evap_cond]
k[vc_adiabatic] = k_vc_adiabatic[vc_adiabatic]
k[wick] = k_wick[wick]
k[wall] = k_wall[wall]

rho[vc_evap_cond] = rho_vc[vc_evap_cond]
rho[vc_adiabatic] = rho_vc[vc_adiabatic]
rho[wick] = rho_wick[wick]
rho[wall] = rho_wall[wall]

c_p[vc_evap_cond] = c_p_vc[vc_evap_cond]
c_p[vc_adiabatic] = c_p_vc[vc_adiabatic]
c_p[wick] = c_p_wick[wick]
c_p[wall] = c_p_wall[wall]

# Calculate D = k / (rho * c_p)
epsilon = 1e-12
D = CellVariable(name="Thermal Diffusivity", mesh=mesh, value=k / (rho * (c_p + epsilon)))

# ----------------------------------------
# Define the PDE
# ----------------------------------------

eq = TransientTerm(var=T) == DiffusionTerm(coeff=D, var=T)

# ----------------------------------------
# Apply boundary conditions
# ----------------------------------------

X, Y = mesh.faceCenters
faces_evaporator = (mesh.facesTop & (X < dimensions['L_e']))
faces_condenser = (mesh.facesTop & ((X > dimensions['L_e'] + dimensions['L_a']) & (X < L_total)))

# preview_face_mask(mesh, faces_evaporator, title="Evaporator Face Mask")
# preview_face_mask(mesh, faces_condenser, title="Condenser Face Mask")

# Define face-normal unit vectors
n = mesh.faceNormals

# FaceVariable version of temperature and conductivity
T_rad = FaceVariable(mesh=mesh, value=T.faceValue)
k_wall_face = FaceVariable(mesh=mesh, value=k_wall)

# Neumann BC at evaporator (heat input)
T.faceGrad.constrain(parameters['Q_input_flux']/10000 * n, where=faces_evaporator)

# Radiative heat loss at condenser
q_rad = constants['sigma'] * parameters['emissivity'] * (T_rad**4 - parameters['T_amb']**4)
T.faceGrad.constrain(-q_rad / 10000 * n, where=faces_condenser)

# ----------------------------------------
# Initialize viewer
# ----------------------------------------

if __name__ == '__main__':
    viewer = Viewer(vars=T)
    viewer.plot()

    # Matplotlib tripcolor viewer:
    # fig, ax, tpc, triang = init_tripcolor_viewer(mesh)

# ----------------------------------------
# Time-stepping loop
# ----------------------------------------

x_point = 0.65  # Specify the point x along the heat pipe (in meters)

# Safe conversion of x_point to index
nx = mesh.shape[0] if hasattr(mesh, 'shape') else all_params.get('nx_vc', 100)
x_index = min(int(x_point / L_total * nx), nx-1)  # Ensure index is within bounds

time_step = all_params['dt']
steps = int(all_params['t_end'] / time_step)

temperature_evolution = []  # List to store temperature values over time
time_values = []  # List to store time values

for step in range(steps):
    try:
        eq.solve(var=T, dt=time_step)
        
        temperature_evolution.append(T.value[x_index])  # Record temperature at x_point
        time_values.append(step * time_step)  # Record the current time
        
        logger.info(f"Step {step+1}/{steps}: T at {x_point} m = {T.value[x_index]} K")
        
        if __name__ == '__main__': 
            try:
                # Mayavi visualization:
                
                viewer.plot()
                
                # Uncomment to save frames (Mayavi)
                # viewer.plot(f"frames/frame_{step:04d}.png")

                # Matplotlib.tripcolor visualization:
                # tpc.set_array(T.value)
                # tpc.set_clim(vmin=T.value.min(), vmax=T.value.max())
                # ax.set_title(f"Step {step+1}, t = {step * time_step:.2f} s")
                # plt.pause(0.003)  # Non-blocking draw
            except Exception as e:
                logger.warning(f"Error during visualization: {e}")
    except Exception as e:
        logger.error(f"Error during time step {step}: {e}")
        break

# Matplotlib.tripcolor visualization:
# plt.close(fig)

# uncomment to save frames (Mayavi)
# save_animation("frames", "output.mp4", fps=10)

# ----------------------------------------
# Plot temperature evolution
# ----------------------------------------

try:
    plt.figure()
    plt.plot(time_values, temperature_evolution, label=f'Temperature at x = {x_point} m')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Evolution Over Time')
    plt.legend()
    plt.grid()
    
    # Save the plot to a file
    output_path = "plots/temperature_evolution"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    
    # Show plot if running interactively
    if __name__ == '__main__':
        plt.show()
except Exception as e:
    logger.error(f"Error generating plot: {e}")
