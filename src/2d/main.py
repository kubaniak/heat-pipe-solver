#!/usr/bin/env python
"""
main.py - Driver script for the 2D heat pipe conduction model using FiPy.
Assumes that functions for meshing, boundary conditions, k_eff computation, etc.,
are provided in separate modules.
"""

from fipy import CellVariable, TransientTerm, DiffusionTerm, Viewer, FaceVariable, ImplicitSourceTerm, Variable, ExplicitDiffusionTerm, GeneralSolver
from fipy.tools import numerix as npx
from fipy.meshes import CylindricalGrid2D, Grid1D
# from tqdm import tqdm
from mesh import generate_mesh_2d, generate_composite_mesh
from params import get_all_params, get_param_group
from utils import preview_mesh, preview_face_mask, save_animation, init_tripcolor_viewer, preview_cell_mask
from material_properties import get_sodium_properties, get_steel_properties, get_vc_properties, get_wick_properties

import matplotlib.pyplot as plt


import logging
log = logging.getLogger("fipy")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
log.addHandler(console)

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

# mesh, cell_types = generate_composite_mesh(mesh_params, dimensions) 
mesh = generate_mesh_2d(L_total, R_total, mesh_params['nx_wall'], mesh_params['nr_wall'])
# Base radial types: vapor core=0, wick=10, wall=20
# Add axial types: evaporator/condenser=+0, adiabatic=+1

x_cell, y_cell = mesh.cellCenters
x_face, y_face = mesh.faceCenters

# preview_mesh(mesh, title="2D Mesh Preview")

# ----------------------------------------
# Define the primary variable (temperature)
# ----------------------------------------

T = CellVariable(name="Temperature", mesh=mesh, value=all_params["T_amb"], hasOld=True)

# ----------------------------------------
# Model validation: simple examples
# ----------------------------------------
# --- Example 1: Stationary heat conduction in a 2D rectangle: convective boundary condition
# mesh = generate_mesh_2d(0.6, 1.0, 500, 500)
# T = CellVariable(name="Temperature", mesh=mesh, value=0.)
# T.constrain(100, where=mesh.facesBottom)
# T_amb = 0
# k = 52
# h = 750
# mask = mesh.facesTop | mesh.facesRight
# Gamma = FaceVariable(mesh=mesh, value=k)
# Gamma.setValue(0., where=mask)
# MA = npx.MA
# tmp = MA.repeat(mesh._faceCenters[..., npx.NewAxis,:], 2, 1)
# cellToFaceDistanceVectors = tmp - npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
# tmp = npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
# tmp = tmp[..., 1,:] - tmp[..., 0,:]
# cellDistanceVectors = MA.filled(MA.where(MA.getmaskarray(tmp), cellToFaceDistanceVectors[:, 0], tmp))
# dPf = FaceVariable(mesh=mesh,
#                    value=mesh._faceToCellDistanceRatio * cellDistanceVectors)
# n = mesh.faceNormals
# a = FaceVariable(mesh=mesh, value=h*n, rank=1)
# b = FaceVariable(mesh=mesh, value=k, rank=0)
# g = FaceVariable(mesh=mesh, value=h*T_amb, rank=0)
# RobinCoeff = mask * k * n / (dPf.dot(a) + b)
# eq = (0 == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff * g).divergence
#        - ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)).divergence, var=T))
# VERDICT: Implementation correct. Note, T.faceGrad.constrain(q_convect*n, where=mesh.facesTop) does NOT work

# --- Example 2: Stationary heat conduction: Ceramic strip (Radiation and Convection)
# T_hot = 900 + 273.15
# T_amb = 50 + 273.15
# k = 3
# h = 50
# epsilon = 0.7

# mesh = generate_mesh_2d(0.02, 0.01, 100*8, 50*8)
# x_cell, y_cell = mesh.cellCenters
# T = CellVariable(name="Temperature", mesh=mesh, value=T_amb)

# T.constrain(T_hot, where=mesh.facesLeft)
# T.constrain(T_hot, where=mesh.facesRight)

# mask = mesh.facesTop
# Gamma = FaceVariable(mesh=mesh, value=k)
# Gamma.setValue(0., where=mask)
# MA = npx.MA
# tmp = MA.repeat(mesh._faceCenters[..., npx.NewAxis,:], 2, 1)
# cellToFaceDistanceVectors = tmp - npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
# tmp = npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
# tmp = tmp[..., 1,:] - tmp[..., 0,:]
# cellDistanceVectors = MA.filled(MA.where(MA.getmaskarray(tmp), cellToFaceDistanceVectors[:, 0], tmp))
# dPf = FaceVariable(mesh=mesh,
#                    value=mesh._faceToCellDistanceRatio * cellDistanceVectors)
# n = mesh.faceNormals
# # Convection
# a_conv = FaceVariable(mesh=mesh, value=h*n, rank=1)
# b_conv = FaceVariable(mesh=mesh, value=k, rank=0)
# g_conv = FaceVariable(mesh=mesh, value=h*T_amb, rank=0)
# RobinCoeff_conv = mask * k * n / (dPf.dot(a_conv) + b_conv)
# eq = (0 == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff_conv * g_conv).divergence
#        - ImplicitSourceTerm(coeff=(RobinCoeff_conv * a_conv.dot(n)).divergence, var=T))

# # Radiation
# a_rad = FaceVariable(mesh=mesh, value=h*n, rank=1)
# b_rad = FaceVariable(mesh=mesh, value=k, rank=0)

# radiation = constants['sigma'] * epsilon * (T_amb**4 - T.faceValue**4)

# g_rad = (h*T_amb + radiation) * mask
# RobinCoeff_rad = mask * k * n / (dPf.dot(a_rad) + b_rad)
# eq = (0 == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff_conv * g_conv).divergence
#        - ImplicitSourceTerm(coeff=(RobinCoeff_conv * a_conv.dot(n)).divergence, var=T)
#        + (RobinCoeff_rad * g_rad).divergence - ImplicitSourceTerm(coeff=(RobinCoeff_rad * a_rad.dot(n)).divergence, var=T))
# VERDICT: Implementation correct. Note, T.faceGrad.constrain(q_radiation*n, where=mesh.facesTop) does NOT work

# --- Example 3: Stationary heat conduction: 2D axisymmetric heat flux boundary condition
# h = 0.5
# r_in = 1.0
# r_out = 2.0
# q = -100.0
# T_out = 10.0
# k = 10.0
# nr = 100.0

# mesh = CylindricalGrid2D(Lz=h, Lr=r_out-r_in, nr=nr, nz=1) + ((r_in,),)
# x_cell, y_cell = mesh.cellCenters
# T = CellVariable(name="Temperature", mesh=mesh, value=T_out)

# T.constrain(T_out, where=mesh.facesRight)
# T.faceGrad.constrain(q/k, where=mesh.facesLeft)

# eq = 0 == DiffusionTerm(coeff=k, var=T)
# VERDICT: Implementation correct. Learned a lot about axisymmetric problems and how to set up the mesh. 
#          Note, T.faceGrad.constrain(q/k, where=mesh.facesLeft) works with either [q/k] or just scalar q/k. 
#          Multiplying by n is not necessary, it just flipped the sign of the flux.

# --- Stationary solver for examples 1-3:
# Choose validation method: 'query_points' or 'analytical'
# validation = 'query_points'
# query_points = [
#     (0.005, 0.0),
#     (0.005, 0.005),
#     (0.01, 0.005),
# ]

# if __name__ == "__main__":
#     viewer = Viewer(vars=T, title="Temperature Distribution")
#     viewer.plot()
# eq.solve()
# if __name__ == "__main__":
#     viewer.plot()
#     if validation == 'query_points':
#         # Validation using query points
#         cell_coords = npx.stack((x_cell, y_cell), axis=1)
#         # Tolerance for proximity (adjust as needed based on mesh size)
#         tolerance = 1e-2
#         print("Queried Temperatures at Specific Points:\n")
#         for point in query_points:
#             distances = npx.linalg.norm(cell_coords - point, axis=1)
#             min_idx = npx.argmin(distances)
#             min_dist = distances[min_idx]
            
#             if min_dist <= tolerance:
#                 temp_value = T.value[min_idx]
#                 coord = cell_coords[min_idx]
#                 print(f"Point {point} found at cell center {tuple(coord)} with T = {temp_value:.4f} K")
#             else:
#                 print(f"Point {point} not found within tolerance {tolerance}. Closest is at {cell_coords[min_idx]} with distance {min_dist:.2e} and T = {T.value[min_idx]:.4f} K")
#     elif validation == 'analytical':
#         # Validation using analytic solution
#         print("Analytical Solution:\n")
#         for i, (x, y) in enumerate(zip(x_cell, y_cell)):
#             analytic_temp = 10 * (1 - npx.log(x / 2))
#             print(f"Cell {i}: x = {x:.4f}, y = {y:.4f}, T_analytic = {analytic_temp:.4f} K, T_numeric = {T.value[i]:.4f} K")

#     from fipy import input
#     input("Press Enter to continue...")

# --- Example 4: Transient 1D heat equation with time-dependent boundary conditions
# t = Variable()
# T_R = 0.0
# T_L = 100.0 * npx.sin(npx.pi*t / 40.0)
# rho = 7200.0
# cp = 440.5
# k = 35.0
# t_end = 32.0 # [s]
# dt = 0.1

# mesh = Grid1D(Lx=0.1, nx=1000)
# T = CellVariable(name="Temperature", mesh=mesh, value=0.)

# T.constrain(T_L, where=mesh.facesLeft)
# T.constrain(T_R, where=mesh.facesRight)

# D = k / (rho * cp)

# eq = (TransientTerm() == DiffusionTerm(coeff=D))

# T.setValue(0.)  # Set initial temperature

# while t() < t_end:
#     t.setValue(t() + dt)
#     eq.solve(var=T, dt=dt)
#     if __name__ == "__main__":
#         viewer.plot()
# from fipy import input
# if __name__ == '__main__':
#     # --- Code to display temperature at specific point ---
#     x_target = 0.02
#     cell_centers_x = mesh.cellCenters[0].value 
#     idx = npx.abs(cell_centers_x - x_target).argmin()
    
#     temp_at_x_target = T.value[idx] # T is a CellVariable, T.value is its numpy array
#     actual_x_at_idx = cell_centers_x[idx]
    
#     print(f"\n--- Results for Example 4 (1D Transient Heat Equation) ---")
#     print(f"Final time t = {t():.1f} s (target t_end = {t_end:.1f} s)") 
#     print(f"Temperature at cell center closest to x = {x_target:.3f} m (actual x = {actual_x_at_idx:.4f} m): {temp_at_x_target:.4f} K")
#     print(f"The expected value from the Wolfram example at x={x_target:.2f}m, t={t_end:.0f}s is approximately 36.6 K.")
#     input("Press <return> to proceed...")
# VERDICT: Implementation correct. Be very careful when setting "hasOld=True" in CellVariable. 
#          It is only needed when sweeping and using the updateOld() method.

# --- Example 5: Transient 1D heat equation: Diffusion within a rod
# T_R = 25
# q_L = 1
# D = 1
# t_end = 0.2
# dt = 0.001

# mesh = Grid1D(Lx=1, nx=1000)
# T = CellVariable(name="Temperature", mesh=mesh, value=25.)

# T.constrain(T_R, where=mesh.facesRight)
# # With faceGrad:
# # T.faceGrad.constrain([q_L], where=mesh.facesLeft)
# # eq = (TransientTerm() == DiffusionTerm(coeff=D))

# # With exterior flux:
# eq = (TransientTerm() == DiffusionTerm(coeff=D) + (mesh.facesLeft * q_L).divergence)

# T.setValue(25.)  # Set initial temperature

# viewer = Viewer(vars=T, title="Temperature Distribution")
# viewer.plot()

# for t in npx.arange(0, t_end, dt):
#     eq.solve(var=T, dt=dt)
#     if __name__ == "__main__":
#         viewer.plot()
# from fipy import input
# if __name__ == '__main__':
#     x = mesh.x
#     T_ref = (24 + x) + 8 / npx.pi**2 * npx.cos(npx.pi/2 * x) * npx.exp(-npx.pi**2 / 4 * t_end) + \
#         8 / 9 / npx.pi**2 * npx.cos(3*npx.pi/2 * x) * npx.exp(-9*npx.pi**2 / 4 * t_end)
#     print(f"\n--- Results for Example 5 (1D Transient Heat Equation) ---")
#     print(f"Comparing numerical and analytical solutions: ")
#     print(npx.allclose(T.value, T_ref, atol=1e-3))
#     input("Press <return> to proceed...")
# VERDICT: Implementation correct. 

# --- Example 6: Transient 2D cylindrical heat conduction with dirichlet boundary conditions
# r = 0.3
# h = 0.4
# T_init = 0.
# T_ext = 1000.
# rho = 7850.
# cp = 460.
# k = 52.
# t_end = 190.
# dt = 0.5

# mesh = CylindricalGrid2D(Lz=h, Lr=r, nr=200, nz=200)
# T = CellVariable(name="Temperature", mesh=mesh, value=T_init)

# T.constrain(T_ext, where=mesh.facesRight | mesh.facesTop | mesh.facesBottom)

# eq = (TransientTerm(coeff=rho*cp) == DiffusionTerm(coeff=k))

# viewer = Viewer(vars=T, title="Temperature Distribution")
# viewer.plot()

# for t in tqdm(npx.arange(0, t_end, dt)):
#     eq.solve(var=T, dt=dt)
#     if __name__ == "__main__":
#         viewer.plot()
# from fipy import input
# if __name__ == '__main__':
#     # Validation using query points
#     cell_coords = npx.stack((mesh.cellCenters[0].value, mesh.cellCenters[1].value), axis=1)
#     query_points = [(0.1, 0.3)]
#     # Tolerance for proximity (adjust as needed based on mesh size)
#     tolerance = 1e-6
#     print("Queried Temperatures at Specific Points:\n")
#     for point in query_points:
#         distances = npx.linalg.norm(cell_coords - point, axis=1)
#         min_idx = npx.argmin(distances)
#         min_dist = distances[min_idx]
        
#         if min_dist <= tolerance:
#             temp_value = T.value[min_idx]
#             coord = cell_coords[min_idx]
#             print(f"Point {point} found at cell center {tuple(coord)} with T = {temp_value:.4f} K")
#         else:
#             print(f"Point {point} not found within tolerance {tolerance}. Closest is at {cell_coords[min_idx]} with distance {min_dist:.2e} and T = {T.value[min_idx]:.4f} K")
# VERDICT: Implementation correct.

# --- Example 7: Stationary 1D heat conduction with temperature-dependent conductivity
# k_0 = 1.0
# T_L = 0.0
# T_R = 100.0
# dt = 0.01
# L = 1.0

# mesh = Grid1D(Lx=L, nx=100)
# T = CellVariable(name="Temperature", mesh=mesh, value=0.0, hasOld=True)

# T.constrain(T_L, where=mesh.facesLeft)
# T.constrain(T_R, where=mesh.facesRight)

# k = k_0 * (1 + T)

# eq = DiffusionTerm(coeff=k)

# if __name__ == "__main__":
#     viewer = Viewer(vars=T, title="Temperature Distribution")
#     viewer.plot()
# T.setValue(0.0)  # Set initial temperature
# res = 1e+10
# while res > 1e-8:
#     res = eq.sweep(var=T, dt=dt)
#     viewer.plot()
#     print(f"Residual: {res:.4e}")

# if __name__ == "__main__":
#     viewer.plot()
#     print("Analytical Solution:\n")
#     x = mesh.x
#     analytical_temp = -1 + npx.sqrt(1 - (x/L * (2 - (T_R+1)**2 +(T_L+1)**2) + (1-(T_L+1)**2)))
#     for i, (x_val, y_val) in enumerate(zip(x, analytical_temp)):
#         print(f"Cell {i}: x = {x_val:.4f}, T_analytic = {y_val:.4f} K, T_numeric = {T.value[i]:.4f} K")
#     from fipy import input
#     input("Press <return> to proceed...")
# VERDICT: Implementation correct. Don't forget to sweep when using temperature-dependent properties.

# --- Example 8: Stationary 1D heat conduction with spatially varying conductivity
# T_L = 0.0
# T_R = 1.0
# L = 1.0

# mesh = Grid1D(Lx=L, nx=100)
# T = CellVariable(name="Temperature", mesh=mesh, value=0.0)

# k = FaceVariable(mesh=mesh, value=1.0)
# x_face = mesh.faceCenters[0]
# k.setValue(0.1, where=(L/4 <= x_face) & (x_face < 3*L/4))

# T.constrain(T_L, where=mesh.facesLeft)
# T.constrain(T_R, where=mesh.facesRight)

# T.setValue(0.0)  # Set initial temperature

# DiffusionTerm(coeff=k).solve(var=T)

# if __name__ == "__main__":
#     viewer = Viewer(vars=T, title="Temperature Distribution")
#     viewer.plot()
#     from fipy import input
#     input("Press <return> to proceed...")
# VERDICT: Implementation correct. Works quite easily

# --- Example 9: Stationary 1D heat conduction with temperature-dependent and spatially varying conductivity
# T_L = 0.0
# T_R = 100.0
# L = 1.0
# k_0 = 1.0
# dt = 0.01

# mesh = Grid1D(Lx=L, nx=1000)
# T = CellVariable(name="Temperature", mesh=mesh, value=0.0)
# x_face = mesh.faceCenters[0]

# k = 1. * ((L / 4. > x_face) | (x_face > 3. * L / 4.)) + k_0 * (1 + T.faceValue) * ((L / 4. <= x_face) & (x_face < 3. * L / 4.))

# T.constrain(T_L, where=mesh.facesLeft)
# T.constrain(T_R, where=mesh.facesRight)

# eq = DiffusionTerm(coeff=k)

# T.setValue(0.0)  # Set initial temperature
# res = 1e+10
# while res > 1e-8:
#     res = eq.sweep(var=T, dt=dt)
#     if __name__ == "__main__":
#         viewer = Viewer(vars=T, title="Temperature Distribution")
#         viewer.plot()
#         print(f"Residual: {res:.4e}")
# if __name__ == "__main__":
#     viewer.plot()
#     from fipy import input
#     input("Press <return> to proceed...")
# VERDICT: I don't know if this is correct. No way to validate. 

# ----------------------------------------
# Define material properties (Tempertaure-dependent!)
# ----------------------------------------

# end_cap_mask = (cell_types == 0) & (x_cell < 0.001) 
# # preview_cell_mask(mesh, end_cap_mask, title="End Cap Mask")

# # Wall properties
# k_wall_cell = steel_properties['thermal_conductivity'](T) # CellTypes 20 and 21
# k_wall_face = steel_properties['thermal_conductivity'](T.harmonicFaceValue) # CellTypes 20 and 21
# c_p_wall_cell = steel_properties['specific_heat'](T) # CellTypes 20 and 21
# c_p_wall_face = steel_properties['specific_heat'](T.harmonicFaceValue) # CellTypes 20 and 21
# rho_wall_evap_adia_cell = steel_properties['density_evap_adia'](T) # CellTypes 20 and 21 and only in the evaporator and adiabatic regions
# rho_wall_evap_adia_face = steel_properties['density_evap_adia'](T.harmonicFaceValue) # CellTypes 20 and 21 and only in the evaporator and adiabatic regions
# rho_wall_cond_cell = steel_properties['density_cond'](T) # CellTypes 20 and 21 and only in the condenser region
# rho_wall_cond_face = steel_properties['density_cond'](T.harmonicFaceValue) # CellTypes 20 and 21 and only in the condenser region

# # Wick properties
# k_wick_cell = wick_properties['thermal_conductivity'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
# k_wick_face = wick_properties['thermal_conductivity'](T.harmonicFaceValue, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
# c_p_wick_cell = wick_properties['specific_heat'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
# c_p_wick_face = wick_properties['specific_heat'](T.harmonicFaceValue, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
# rho_wick_cell = wick_properties['density'](T, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11
# rho_wick_face = wick_properties['density'](T.harmonicFaceValue, sodium_properties, steel_properties, parameters) # CellTypes 10 and 11

# # Vapor core properties
# k_vc_evap_cond_cell = vc_properties['thermal_conductivity'](T, mesh, 'evap_cond', sodium_properties, dimensions, parameters, constants) # CellType 0
# k_vc_evap_cond_face = vc_properties['thermal_conductivity'](T.harmonicFaceValue, mesh, 'evap_cond', sodium_properties, dimensions, parameters, constants) # CellType 0
# k_vc_adiabatic_cell = vc_properties['thermal_conductivity'](T, mesh, 'adiabatic', sodium_properties, dimensions, parameters, constants) # CellType 1
# k_vc_adiabatic_face = vc_properties['thermal_conductivity'](T.harmonicFaceValue, mesh, 'adiabatic', sodium_properties, dimensions, parameters, constants) # CellType 1
# c_p_vc_cell = vc_properties['specific_heat'](T) # CellTypes 0 and 1
# c_p_vc_face = vc_properties['specific_heat'](T.harmonicFaceValue) # CellTypes 0 and 1
# rho_vc_cell = vc_properties['density'](T) # CellTypes 0 and 1
# rho_vc_face = vc_properties['density'](T.harmonicFaceValue) # CellTypes 0 and 1

# ----------------------------------------
# Define the spatially varying D coefficient 
# ----------------------------------------

# # Convenience masks
# vc_evap_cond_cells = (cell_types == 0)
# vc_adiabatic_cells = (cell_types == 1)
# vc_cells = vc_evap_cond_cells | vc_adiabatic_cells
# wick_cells = (cell_types == 10) | (cell_types == 11)
# wall_cells = (cell_types == 20) | (cell_types == 21)
# wall_cond_cells = wall_cells & (x_cell > dimensions['L_e'] + dimensions['L_a'])
# wall_evap_adia_cells = wall_cells & (x_cell < dimensions['L_e'] + dimensions['L_a']) # Wall in evaporator and adiabatic regions

# preview_cell_mask(mesh, vc_evap_cond, title="Vapor Core Evaporator and Condenser Region")
# preview_cell_mask(mesh, vc_adiabatic, title="Vapor Core Adiabatic Region")
# preview_cell_mask(mesh, wick, title="Wick Region")
# preview_cell_mask(mesh, wall, title="Wall Region")
# preview_cell_mask(mesh, wall_cond, title="Wall Condenser Region")
# preview_cell_mask(mesh, wall_evap_adia, title="Wall Evaporator and Adiabatic Region")

# vc_faces = (dimensions['R_vc'] >= y_face)
# vc_evap_cond_faces = vc_faces & ((x_face < dimensions['L_e']) | (x_face > (dimensions['L_e'] + dimensions['L_a'])))
# vc_adiabatic_faces = vc_faces & ((x_face > (dimensions['L_e'])) & (x_face < dimensions['L_e'] + dimensions['L_a']))
# wick_faces = (dimensions['R_vc'] < y_face) & (y_face < dimensions['R_wick'])
# wall_faces = dimensions['R_wick'] < y_face
# wall_evap_adia_faces = wall_faces & (x_face < (dimensions['L_e'] + dimensions['L_a']))
# wall_cond_faces = wall_faces & (x_face > (dimensions['L_e'] + dimensions['L_a']))

# preview_face_mask(mesh, wall_faces, title="Wall Faces")
# preview_face_mask(mesh, wall_evap_adia_faces, title="Wall Evaporator and Adiabatic Faces")
# preview_face_mask(mesh, wall_cond_faces, title="Wall Condenser Faces")
# preview_face_mask(mesh, wick_faces, title="Wick Faces")
# preview_face_mask(mesh, vc_faces, title="Vapor Core Faces")
# preview_face_mask(mesh, vc_evap_cond_faces, title="Vapor Core Evaporator and Condenser Faces")
# preview_face_mask(mesh, vc_adiabatic_faces, title="Vapor Core Adiabatic Faces")

# # Calculate D = k / (rho * c_p)
# D_expr = 0 * T
# epsilon = 1e-12

# # Add symbolic contributions by region
# D_expr = D_expr + ((cell_types == 0) * (k_vc_evap_cond_cell / (rho_vc_cell * c_p_vc_cell + epsilon)))
# D_expr = D_expr + ((cell_types == 1) * (k_vc_adiabatic_cell / (rho_vc_cell * c_p_vc_cell + epsilon)))
# D_expr = D_expr + ((wick_cells) * (k_wick_cell / (rho_wick_cell * c_p_wick_cell + epsilon)))
# D_expr = D_expr + ((wall_cond_cells) * (k_wall_cell / (rho_wall_cond_cell * c_p_wall_cell + epsilon)))
# D_expr = D_expr + ((wall_evap_adia_cells) * (k_wall_cell / (rho_wall_evap_adia_cell * c_p_wall_cell + epsilon)))

# D_var = CellVariable(name="Diffusivity", mesh=mesh, value=D_expr, hasOld=True)

# # ----------------------------------------
# # Define CellVariables for k, rho, c_p
# # ----------------------------------------
# k_expr = 0 * T
# k_expr = k_expr + (vc_evap_cond_cells * k_vc_evap_cond_cell)
# k_expr = k_expr + (vc_adiabatic_cells * k_vc_adiabatic_cell)
# k_expr = k_expr + (wick_cells * k_wick_cell)
# k_expr = k_expr + (wall_cells * k_wall_cell) # k_wall is the same for wall_cond and wall_evap_adia
# k_var = CellVariable(name="ThermalConductivity", mesh=mesh, value=k_expr, hasOld=True)

# rho_expr = 0 * T
# rho_expr = rho_expr + ((vc_evap_cond_cells | vc_adiabatic_cells) * rho_vc_cell) # rho_vc is same for both
# rho_expr = rho_expr + (wick_cells * rho_wick_cell)
# rho_expr = rho_expr + (wall_cond_cells * rho_wall_cond_cell)
# rho_expr = rho_expr + (wall_evap_adia_cells * rho_wall_evap_adia_cell)
# rho_var = CellVariable(name="Density", mesh=mesh, value=rho_expr, hasOld=True)

# cp_expr = 0 * T
# cp_expr = cp_expr + ((vc_evap_cond_cells | vc_adiabatic_cells) * c_p_vc_cell) # c_p_vc is same for both
# cp_expr = cp_expr + (wick_cells * c_p_wick_cell)
# cp_expr = cp_expr + (wall_cells * c_p_wall_cell) # c_p_wall is same for both
# cp_var = CellVariable(name="SpecificHeat", mesh=mesh, value=cp_expr, hasOld=True)

# D_expr = D_expr + (k_expr / (rho_expr * cp_expr + epsilon))
# D_var = k_var / (rho_var * cp_var + epsilon)

# ----------------------------------------
# Apply boundary conditions
# ----------------------------------------

# Define masks
faces_evaporator = (mesh.facesTop & ((x_face < dimensions['L_input_right']) & (x_face > dimensions['L_input_left'])))
faces_condenser = (mesh.facesTop & (x_face > (dimensions['L_e'] + dimensions['L_a'])) & (x_face < L_total))

# constants
k = 5.0
sigma_b = constants['sigma']
T_amb = parameters['T_amb']
h = 750.0

# # Constant influx
q = parameters['Q_input_flux']
# T.faceGrad.constrain(q / k, where=faces_evaporator)

# Convective boundary condition
Gamma = FaceVariable(mesh=mesh, value=k)
Gamma.setValue(0., where=faces_condenser)
MA = npx.MA
tmp = MA.repeat(mesh._faceCenters[..., npx.NewAxis,:], 2, 1)
cellToFaceDistanceVectors = tmp - npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
tmp = npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
tmp = tmp[..., 1,:] - tmp[..., 0,:]
cellDistanceVectors = MA.filled(MA.where(MA.getmaskarray(tmp), cellToFaceDistanceVectors[:, 0], tmp))
dPf = FaceVariable(mesh=mesh,
                   value=mesh._faceToCellDistanceRatio * cellDistanceVectors)
n = mesh.faceNormals
a = FaceVariable(mesh=mesh, value=h*n, rank=1)
b = FaceVariable(mesh=mesh, value=k, rank=0)
g = FaceVariable(mesh=mesh, value=h*T_amb, rank=0)
RobinCoeff = faces_condenser * k * n / (dPf.dot(a) + b)
eq = (TransientTerm(var=T) == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff * g).divergence
       - ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)).divergence, var=T)
       + (faces_evaporator * (q/k)).divergence)

T.setValue(T_amb)  # Set initial temperature

# ----------------------------------------
# Time-stepping loop
# ----------------------------------------

dt = all_params['dt']
t_end = 15.0

viewer = Viewer(vars=T, title="Temperature Distribution")
viewer.plot()

for t in tqdm(npx.arange(0, t_end, dt)):
    eq.solve(var=T, dt=dt)
    if __name__ == "__main__":
        viewer.plot()


# # Variable to plot alongside Temperature. Options: "D_var", "k_plot_var", "rho_plot_var", "cp_plot_var", "T"
# plot_selection = "T" # CHANGE THIS STRING TO VISUALIZE OTHER VARIABLES

# plottable_vars_map = {
#     "D_var": D_var,
#     "k_var": k_var,
#     "rho_var": rho_var,
#     "cp_var": cp_var,
#     "T": T
# }
# selected_aux_var = plottable_vars_map[plot_selection]

# if __name__ == "__main__":
#     if plot_selection == "T":
#         viewer = Viewer(vars=T, title="Temperature Distribution")
#     else:
#         viewer = Viewer(vars=(T, selected_aux_var), title=f"Temperature and {selected_aux_var.name} Distribution")

# measure_times = [0, 1, 2, 10, 30, 40, 100, 150, 400]  # Time points to measure temperature (t / dt)

# # Find topmost wall cells: any face of the cell is a top face
# cellFaceIDs = npx.array(mesh.cellFaceIDs)  # shape: (nFacesPerCell, nCells)
# facesTop = npx.array(mesh.facesTop)        # shape: (nFaces,)

# # For each cell, check if any of its faces are top faces
# # top_wall_mask = wall_cells & npx.any(facesTop[cellFaceIDs], axis=0)

# # preview_cell_mask(mesh, top_wall_mask, title="Top Wall Cells")

# # Prepare to store temperature profiles at measure_times for top wall cells
# profiles = []
# profile_times = []

# # Setting up timestepping
# dt = all_params['dt']
# timestep = 0
# t_end = all_params['t_end']
# t = timestep * dt

from fipy.solvers import LinearGMRESSolver # Changed from LinearLUSolver
from fipy.solvers import LinearLUSolver # Original solver, commented out
# from fipy.solvers import LinearBICGSolver # Another alternative iterative solver
# from fipy.preconditioners import PointJacobiPreconditioner # Example if a preconditioner is needed

solver = LinearLUSolver(tolerance=1e-6, iterations=1000) # Original solver
solver = LinearGMRESSolver(tolerance=1e-6, iterations=1000) # Using GMRES
# Or, try LinearBICGSolver:
# solver = LinearBICGSolver(tolerance=1e-6, iterations=1000)

# T.setValue(all_params["T_amb"])  # Set initial temperature
# viewer.plot()  # Initial plot

# import time
# time.sleep(3) # Pause to allow viewer to initialize

# viewer = Viewer(vars=T, title="Temperature Distribution")
# viewer.plot()

# for t in npx.arange(0, t_end, dt):
#     eq.solve(var=T, dt=dt)
#     if __name__ == "__main__":
#         viewer.plot()
# from fipy import input
# input("Press <return> to proceed...")
    

while t < t_end:
    t += dt
    timestep += 1
    T.updateOld()
    # D_var.setValue(D_expr) # Explicitly update D_var after T changes
    res = 1e+10
    # if timestep in measure_times:
    #     profiles.append(T.value[top_wall_mask].copy())
    #     profile_times.append(timestep * dt)
    while res > 1e-4:
        # print(f"Current T min: {T.min()}, T max: {T.max()}")
        res = eq.sweep(var=T, dt=dt, solver=solver)
        # print(f"Sweep Time: {t:.2f} / {t_end:.2f} s --- Residual: {res}")
    # if timestep % 10 == 0:
    #     if __name__ == "__main__":
    #         viewer.plot()
    print(f"STEP! Time: {t:.2f} / {t_end:.2f} s")

# Get x-coordinates for top wall cells
# x_top_wall = x_cell[top_wall_mask]

# # Sort by x for plotting
# sort_idx = x_top_wall.argsort()
# x_top_wall_sorted = x_top_wall[sort_idx]

# plt.figure(figsize=(10, 6))
# for i, (T_profile, t) in enumerate(zip(profiles, profile_times)):
#     plt.plot(x_top_wall_sorted, T_profile[sort_idx], label=f"t = {t:.2f} s")
# plt.xlabel("x [m]")
# plt.ylabel("Temperature [K]")
# plt.title("Axial Temperature Profile in Topmost Wall Cells at Different Times")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/top_wall_axial_profiles.png", dpi=300)
# plt.show()