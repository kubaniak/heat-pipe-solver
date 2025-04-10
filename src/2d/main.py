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
# from k_eff import getKEff 
# from sodium_properties import get_sodium_properties
# from postprocess import save_results, plot_results


# ----------------------------------------
# Load parameters from configuration file
# ----------------------------------------

params = get_all_params()
mesh_params = get_param_group('mesh')
dimensions = get_param_group('dimensions')

L_total = params['L_e'] + params['L_a'] + params['L_c']
R_total = params['R_wall'] + params['R_wick'] + params['R_vc']

# ----------------------------------------
# Generate the 2D mesh
# ----------------------------------------

mesh, cell_types = generate_composite_mesh(mesh_params, dimensions)

# preview the mesh (optional)
# preview_mesh(mesh, title="2D Mesh Preview")

# ----------------------------------------
# Define the primary variable (temperature)
# ----------------------------------------

T = CellVariable(name="Temperature", mesh=mesh, value=params["T_amb"])

# ----------------------------------------
# Define the effective thermal conductivity variable (will do later, now use constant k)
# ----------------------------------------
# k = params["k"] # Constant thermal conductivity for now
# getKEff(T, params) returns a FiPy-compatible array (or CellVariable value)
# k_eff = CellVariable(name="Effective Thermal Conductivity", mesh=mesh, value=getKEff(T, params))

# ----------------------------------------
# Define the PDE
# ----------------------------------------

eq = TransientTerm() == DiffusionTerm(coeff=1e-3)

# ----------------------------------------
# Apply boundary conditions
# ----------------------------------------

X, Y = mesh.faceCenters
faces_evaporator = (mesh.facesTop & (X < params['L_e']))
faces_condenser = (mesh.facesTop & ((X > params['L_e'] + params['L_a']) & (X < L_total)))

# Mask previewing (optional)
# preview_face_mask(mesh, faces_evaporator, title="Evaporator Face Mask")
# preview_face_mask(mesh, faces_condenser, title="Condenser Face Mask")

n = mesh.faceNormals
T.faceGrad.constrain(params['Q_input_flux']/params['k_wall'] * n, where=faces_evaporator)  # dT/dy = q_e/k_w (condenser heat input)

T_rad = FaceVariable(mesh=mesh, value=T.faceValue)  # Radiative temperature at the faces
q_rad = params['sigma'] * params['emissivity'] * (T_rad**4 - params['T_amb']**4)  # Radiative heat flux (W/m^2)
T.faceGrad.constrain(-q_rad / params['k_wall'] * n, where=faces_condenser)  # dT/dy = -q_rad/k_w

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
nx = mesh.shape[0] if hasattr(mesh, 'shape') else params.get('nx_vc', 100)
x_index = min(int(x_point / L_total * nx), nx-1)  # Ensure index is within bounds

time_step = params['dt']
steps = int(params['t_end'] / time_step)

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
