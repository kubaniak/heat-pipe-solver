#!/usr/bin/env python
"""
main.py - Driver script for the 2D heat pipe conduction model using FiPy.
Assumes that functions for meshing, boundary conditions, k_eff computation, etc.,
are provided in separate modules.
"""

import numpy as np
from fipy import CellVariable, TransientTerm, DiffusionTerm, Viewer, FaceVariable
from mesh import generate_mesh_2d
from params import get_all_params
# from k_eff import getKEff 
# from sodium_properties import get_sodium_properties
# from postprocess import save_results, plot_results

from utils import preview_mesh, preview_face_mask

# ----------------------------------------
# Load parameters from configuration file
# ----------------------------------------

params = get_all_params()
L_total = params['L_e'] + params['L_a'] + params['L_c']
R_total = params['R_wall'] + params['R_wick'] + params['R_vc']

# ----------------------------------------
# Generate the 2D mesh
# ----------------------------------------

mesh = generate_mesh_2d(L_total, R_total, params["nx_vc"], params["nr_wall"])
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
# Using FiPy's PDE syntax:
#    TransientTerm(rho*c_p)*T == DiffusionTerm(coeff=k_eff)

eq = TransientTerm() == DiffusionTerm(coeff=1e-3)

# ----------------------------------------
# Apply boundary conditions (to be refined later)
# ----------------------------------------

X, Y = mesh.faceCenters
faces_evaporator = (mesh.facesTop & (X < params['L_e']))
faces_condenser = (mesh.facesTop & ((X > params['L_e'] + params['L_a']) & (X < L_total)))

preview_face_mask(mesh, faces_evaporator, title="Evaporator Face Mask")
# preview_face_mask(mesh, faces_condenser, title="Condenser Face Mask")

n = mesh.faceNormals
T.faceGrad.constrain(params['Q_input_flux']/params['k_wall'] * n, where=faces_evaporator)  # dT/dy = q_e/k_w (condenser heat input)

T_rad = FaceVariable(mesh=mesh, value=T.faceValue)  # Radiative temperature at the faces
q_rad = params['sigma'] * params['emissivity'] * (T_rad**4 - params['T_amb']**4)  # Radiative heat flux (W/m^2)
T.faceGrad.constrain(-q_rad / params['k_wall'] * n, where=faces_condenser)  # dT/dy = -q_rad/k_w

# ----------------------------------------
# Define the solver
# ----------------------------------------

if __name__ == '__main__':
    viewer = Viewer(vars=T, datamin=0., datamax=1.)
    viewer.plot()

# ----------------------------------------
# Time-stepping loop
# ----------------------------------------

time_step = params['dt']
steps = int(params['t_end'] / time_step)



for step in range(steps):
    eq.solve(var=T,
             dt=time_step)
    
    print(f"Step {step+1}/{steps}: T at 0.65 m = {T.value[int(0.65 / L_total * params['nx_vc'])]} K")
    if __name__ == '__main__':
        viewer.plot()