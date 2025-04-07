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

def main():
    # ----------------------------------------
    # Load parameters from configuration file
    # ----------------------------------------
    params = get_all_params()
    L_total = params['L_e'] + params['L_a'] + params['L_c']
    R_total = params['R_wall'] + params['R_wick'] + params['R_vc']
    # ----------------------------------------
    # Generate the 2D mesh
    # ----------------------------------------
    # generate_mesh_2d(L_x, L_y, nx, ny) should return a FiPy mesh (e.g., a Grid2D)
    mesh = generate_mesh_2d(L_total, R_total, params["nx_vc"], params["nr_wall"])
    
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

    D = 1
    eq = TransientTerm() == DiffusionTerm(coeff=D)
    
    # ----------------------------------------
    # Apply initial boundary conditions (to be refined later)
    # ----------------------------------------
    
    X, Y = mesh.faceCenters
    faces_evaporator = (mesh.facesTop & (X < params['L_e']))
    faces_condenser = (mesh.facesTop & ((X > params['L_e'] + params['L_a']) & (X < L_total)))

    n = mesh.faceNormals
    T.faceGrad.constrain(params['Q_input_flux']/params['k_wall'] * n, where=faces_evaporator)  # dT/dy = q_e/k_w (condenser heat input)

    T_rad = FaceVariable(mesh=mesh, value=T.faceValue)  # Radiative temperature at the faces
    q_rad = params['sigma'] * params['emissivity'] * (T_rad**4 - params['T_amb']**4)  # Radiative heat flux (W/m^2)
    T.faceGrad.constrain(-q_rad / params['k_wall'] * n, where=faces_condenser)  # dT/dy = -q_rad/k_w
    
    # ----------------------------------------
    # Setup visualization (optional)
    # ----------------------------------------
    viewer = Viewer(vars=T)
    
    # ----------------------------------------
    # Time stepping and simulation
    # ----------------------------------------
    n_steps = int(params["t_end"] / params["dt"])
    # Preallocate a list to store temperature history at each time step
    T_history = []
    T_history.append(T.value.copy())
    
    for step in range(n_steps):
        # Update k_eff based on the current temperature field
        # This update function is assumed to incorporate the sonic limit.
        # k_eff.setValue(getKEff(T, params))
        
        # Solve the PDE for one time step
        eq.solve(var=T, dt=params["dt"])
        
        # Store current temperature field in history
        T_history.append(T.value.copy())
        
        # Optional visualization every 100 steps
        if step % 100 == 0:
            viewer.plot()

if __name__ == '__main__':
    main()
