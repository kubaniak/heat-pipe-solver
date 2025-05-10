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

T = CellVariable(name="Temperature", mesh=mesh, value=all_params["T_amb"], hasOld=True)

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

# end_cap_mask = (cell_types == 0) & (x_cell < 0.001) 

# preview_cell_mask(mesh, vc_evap_cond, title="Vapor Core Evaporator and Condenser Region")
# preview_cell_mask(mesh, vc_adiabatic, title="Vapor Core Adiabatic Region")
# preview_cell_mask(mesh, wick, title="Wick Region")
# preview_cell_mask(mesh, wall, title="Wall Region")
# preview_cell_mask(mesh, wall_cond, title="Wall Condenser Region")
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

D_var = CellVariable(name="Diffusivity", mesh=mesh, value=D_expr, hasOld=True)

# ----------------------------------------
# Define CellVariables for k, rho, c_p for plotting
# ----------------------------------------
k_expr = 0 * T
k_expr = k_expr + (vc_evap_cond * k_vc_evap_cond)
k_expr = k_expr + (vc_adiabatic * k_vc_adiabatic)
k_expr = k_expr + (wick * k_wick)
k_expr = k_expr + (wall * k_wall) # k_wall is the same for wall_cond and wall_evap_adia
k_plot_var = CellVariable(name="ThermalConductivity", mesh=mesh, value=k_expr, hasOld=True)

rho_expr = 0 * T
rho_expr = rho_expr + ((vc_evap_cond | vc_adiabatic) * rho_vc) # rho_vc is same for both
rho_expr = rho_expr + (wick * rho_wick)
rho_expr = rho_expr + (wall_cond * rho_wall_cond)
rho_expr = rho_expr + (wall_evap_adia * rho_wall_evap_adia)
rho_plot_var = CellVariable(name="Density", mesh=mesh, value=rho_expr, hasOld=True)

cp_expr = 0 * T
cp_expr = cp_expr + ((vc_evap_cond | vc_adiabatic) * c_p_vc) # c_p_vc is same for both
cp_expr = cp_expr + (wick * c_p_wick)
cp_expr = cp_expr + (wall * c_p_wall) # c_p_wall is same for both
cp_plot_var = CellVariable(name="SpecificHeat", mesh=mesh, value=cp_expr, hasOld=True)

# ----------------------------------------
# Define the PDE
# ----------------------------------------

eq = TransientTerm(var=T) == DiffusionTerm(coeff=D_expr, var=T)

# ----------------------------------------
# Apply boundary conditions
# ----------------------------------------

x_face, y_face = mesh.faceCenters
faces_evaporator = (mesh.facesTop & ((x_face < dimensions['L_input_right']) & (x_face > dimensions['L_input_left'])))
faces_condenser = (mesh.facesTop & (x_face > (dimensions['L_e'] + dimensions['L_a'])) & (x_face < L_total))

# cells_evaporator = (cell_types == 0) & (x_cell < dimensions['L_input_right']) & (x_cell > dimensions['L_input_left']) # Not directly used for this BC
# cells_condenser = (cell_types == 0) & ((x_cell > dimensions['L_e'] + dimensions['L_a']) & (x_cell < L_total)) # Not used

# preview_face_mask(mesh, faces_evaporator, title="Evaporator Face Mask")
# preview_face_mask(mesh, faces_condenser, title="Condenser Face Mask") # Condenser not used

# Define face-normal unit vectors
n = mesh.faceNormals

# Calculate the volumetric heat source from the input flux (W/m^3)
volumetric_heat_source_W_m3 = (faces_evaporator * parameters['Q_input_flux'] * n).divergence

# Get the rho * c_p for the wall cells in the evaporator region
# These are CellVariables and will be evaluated cell by cell where the source is applied.
# rho_wall_evap_adia and c_p_wall are already defined as CellVariable functions of T
rho_cp_wall_evaporator = rho_wall_evap_adia * c_p_wall

# Add a small epsilon to prevent division by zero if rho_cp could be zero.
eps = 1e-12 

# Convert the volumetric hvolumetric_heat_source_W_m3eat source to K/s
source_term_K_s =  volumetric_heat_source_W_m3 / (rho_cp_wall_evaporator + eps)

# Re-define eq with the correctly scaled evaporator flux source term
eq = TransientTerm(var=T) == (DiffusionTerm(coeff=D_expr, var=T) + volumetric_heat_source_W_m3)


# ------------------------------------------
# Radiative boundary condition (Neumann) with sink term
# ------------------------------------------

# # Apply the radiative boundary condition at the condenser faces
# sigma_sb = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4) - Corrected value
# epsilon_rad = parameters['emissivity']
# # T_amb is available in all_params['T_amb']

# # Define condenser faces (top faces in the condenser section)
# # x_face, L_total, dimensions['L_e'], dimensions['L_a'] are already defined

# # The radiative heat flux is q_rad = sigma_sb * epsilon_rad * (T_surface^4 - T_ambient^4)
# # The boundary condition is -k_wall * (dT/dn) = q_rad
# # So, dT/dn = -q_rad / k_wall
# # Ensure all terms are evaluated at the faces for the constraint:
# T_face = T.arithmeticFaceValue  # Temperature at the faces
# k_wall_at_face = steel_properties['thermal_conductivity'](T_face)  # Wall thermal conductivity at the faces

# radiative_numerator_at_face = sigma_sb * epsilon_rad * (T_face**4 - all_params['T_amb']**4)

# # k_wall (CellVariable) is defined earlier. We use k_wall_at_face (FaceVariable) here.

# constraint_value_radiation_at_face = -radiative_numerator_at_face / (k_wall_at_face + epsilon_k_denom)
# T.faceGrad.constrain(constraint_value_radiation_at_face, where=faces_condenser)

# volumetric_heat_sink_W_m3 = (faces_condenser * parameters['Q_output_flux'] * n).divergence

# rho_cp_wall_condenser = rho_wall_cond * c_p_wall

# sink_term_K_s = volumetric_heat_sink_W_m3 / (rho_cp_wall_condenser + epsilon_source_denom)

# eq = TransientTerm(var=T) == (DiffusionTerm(coeff=D_expr, var=T) + source_term_K_s + sink_term_K_s)


# ------------------------------------------
# Convection boundary condition (Robin)
# ------------------------------------------

# Try with convective boundary condition (Robin)
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm

# Define where to apply the Robin BC
faces_condenser = (mesh.facesTop & (x_face > (dimensions['L_e'] + dimensions['L_a'])) & (x_face < L_total))

# Mask to apply only on those faces
mask = FaceVariable(mesh=mesh, value=0.)
mask.setValue(1., where=faces_condenser)

# Heat transfer coefficient and ambient temperature
h = 25  # [W/m²K], given
T_amb = parameters['T_amb']  # Ambient temperature

# Gamma and normal vector
Gamma = FaceVariable(mesh=mesh, value=D_expr)  # temperature-dependent diffusivity
Gamma.setValue(0., where=~faces_condenser)     # deactivate diffusion on non-condenser faces

dPf = FaceVariable(mesh=mesh, value=mesh._faceToCellDistanceRatio * mesh.cellDistanceVectors)
n = mesh.faceNormals

# Robin BC terms
a = FaceVariable(mesh=mesh, value=h * n, rank=1)       # a = h * n
b = FaceVariable(mesh=mesh, value=k_expr.arithmeticFaceValue, rank=0)      # b = k(x), spatially varying
g = FaceVariable(mesh=mesh, value=h * T_amb, rank=0)   # g = h * T_amb

# Robin coefficient
RobinCoeff = mask * D_expr.arithmeticFaceValue * n / (dPf.dot(a) + b)

# Final equation with Robin boundary condition
eqn = (TransientTerm(var=T) ==
       DiffusionTerm(coeff=Gamma, var=T)
       + source_term_K_s
       + (RobinCoeff * g).divergence
       - ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)).divergence))


# ----------------------------------------
# guyer implementation for radiative boundary condition
# ----------------------------------------

# q_rad = constants['sigma'] * parameters['emissivity']  * (T.faceValue**4 - parameters['T_amb']**4)

# rho_cp_wall_evaporator = rho_wall_evap_adia * c_p_wall
# rho_cp_wall_condenser = rho_wall_cond * c_p_wall

# source_term_K_s = (((faces_evaporator * parameters['Q_input_flux'] * n).divergence) / (rho_cp_wall_evaporator + eps))
# sink_term_K_s = (((faces_condenser * q_rad * n).divergence) / (rho_cp_wall_condenser + eps))

# eq = TransientTerm(var=T) == (DiffusionTerm(coeff=D_var, var=T)
#                               + source_term_K_s
#                               + sink_term_K_s)


# ----------------------------------------
# Time-stepping loop
# ----------------------------------------

# Variable to plot alongside Temperature. Options: "D_var", "k_plot_var", "rho_plot_var", "cp_plot_var"
plot_selection = "T" # CHANGE THIS STRING TO VISUALIZE OTHER VARIABLES

plottable_vars_map = {
    "D_var": D_var,
    "k_plot_var": k_plot_var,
    "rho_plot_var": rho_plot_var,
    "cp_plot_var": cp_plot_var,
    "T": T
}
selected_aux_var = plottable_vars_map[plot_selection]

# if __name__ == "__main__":
#     if plot_selection == "T":
#         viewer = Viewer(vars=T, title="Temperature Distribution")
#     else:
#         viewer = Viewer(vars=(T, selected_aux_var), title=f"Temperature and {selected_aux_var.name} Distribution")

measure_times = [300, 400, 450]  # Time points to measure temperature (t / dt)

# Find topmost wall cells: any face of the cell is a top face
cellFaceIDs = npx.array(mesh.cellFaceIDs)  # shape: (nFacesPerCell, nCells)
facesTop = npx.array(mesh.facesTop)        # shape: (nFaces,)

# For each cell, check if any of its faces are top faces
top_wall_mask = wall & npx.any(facesTop[cellFaceIDs], axis=0)

# preview_cell_mask(mesh, top_wall_mask, title="Top Wall Cells")

# Prepare to store temperature profiles at measure_times for top wall cells
profiles = []
profile_times = []

# Setting up timestepping
dt = all_params['dt']
timestep = 0
run_time = all_params['t_end']
t = timestep * dt

from fipy.solvers import LinearLUSolver
solver = LinearLUSolver(tolerance=1e-8, iterations=1000)

while t < run_time:
    t += dt
    timestep += 1
    T.updateOld()
    D_var.setValue(D_expr) # Explicitly update D_var after T changes
    # k_plot_var.setValue(k_expr)
    # rho_plot_var.setValue(rho_expr)
    # cp_plot_var.setValue(cp_expr)
    res = 1e+10
    if timestep in measure_times:
        profiles.append(T.value[top_wall_mask].copy())
        profile_times.append(timestep * dt)
    while res > 1e-4:
        # print(f"Current T min: {T.min()}, T max: {T.max()}")
        res = eq.sweep(dt=dt, solver=solver)
    # if __name__ == "__main__":
    #     viewer.plot()
    print(f"Time: {t:.2f} / {run_time:.2f} s")

# Get x-coordinates for top wall cells
x_top_wall = x_cell[top_wall_mask]

# Sort by x for plotting
sort_idx = x_top_wall.argsort()
x_top_wall_sorted = x_top_wall[sort_idx]

plt.figure(figsize=(10, 6))
for i, (T_profile, t) in enumerate(zip(profiles, profile_times)):
    plt.plot(x_top_wall_sorted, T_profile[sort_idx], label=f"t = {t:.2f} s")
plt.xlabel("x [m]")
plt.ylabel("Temperature [K]")
plt.title("Axial Temperature Profile in Topmost Wall Cells at Different Times")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/top_wall_axial_profiles.png", dpi=300)
plt.show()