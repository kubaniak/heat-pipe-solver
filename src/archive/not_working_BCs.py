# x_face, y_face = mesh.faceCenters
# faces_evaporator = (mesh.facesTop & ((x_face < dimensions['L_input_right']) & (x_face > dimensions['L_input_left'])))
# faces_condenser = (mesh.facesTop & (x_face > (dimensions['L_e'] + dimensions['L_a'])) & (x_face < L_total))

# cells_evaporator = (cell_types == 0) & (x_cell < dimensions['L_input_right']) & (x_cell > dimensions['L_input_left']) # Not directly used for this BC
# cells_condenser = (cell_types == 0) & ((x_cell > dimensions['L_e'] + dimensions['L_a']) & (x_cell < L_total)) # Not used

# preview_face_mask(mesh, faces_evaporator, title="Evaporator Face Mask")
# preview_face_mask(mesh, faces_condenser, title="Condenser Face Mask") # Condenser not used

# Define face-normal unit vectors
# n = mesh.faceNormals

# Calculate the volumetric heat source from the input flux (W/m^3)
# volumetric_heat_source_W_m3 = (faces_evaporator * parameters['Q_input_flux'] * n).divergence

# # Get the rho * c_p for the wall cells in the evaporator region
# # These are CellVariables and will be evaluated cell by cell where the source is applied.
# # rho_wall_evap_adia and c_p_wall are already defined as CellVariable functions of T
# rho_cp_wall_evaporator = rho_wall_evap_adia_cell * c_p_wall_cell

# Add a small epsilon to prevent division by zero if rho_cp could be zero.
# eps = 1e-12 

# # Convert the volumetric hvolumetric_heat_source_W_m3eat source to K/s
# source_term_K_s = volumetric_heat_source_W_m3 / (rho_cp_wall_evaporator + eps)

# # Re-define eq with the correctly scaled evaporator flux source term
# eq = TransientTerm(var=T) == (DiffusionTerm(coeff=D_expr, var=T) + source_term_K_s)

# ------------------------------------------
# Convection boundary condition (Robin) (not working)
# ------------------------------------------

# # Try with convective boundary condition (Robin)
# from fipy.terms.implicitSourceTerm import ImplicitSourceTerm

# # Define where to apply the Robin BC
# faces_condenser = (mesh.facesTop & (x_face > (dimensions['L_e'] + dimensions['L_a'])) & (x_face < L_total))

# # Mask to apply only on those faces
# mask = FaceVariable(mesh=mesh, value=0.)
# mask.setValue(1., where=faces_condenser)

# # Heat transfer coefficient and ambient temperature
# h = 25  # [W/m²K], given
# T_amb = parameters['T_amb']  # Ambient temperature

# # Gamma and normal vector
# Gamma = FaceVariable(mesh=mesh, value=D_expr)  # temperature-dependent diffusivity
# Gamma.setValue(0., where=faces_condenser)     # deactivate diffusion on non-condenser faces

# dPf = FaceVariable(mesh=mesh, value=mesh._faceToCellDistanceRatio * mesh.cellDistanceVectors)
# n = mesh.faceNormals

# # Robin BC terms
# a = FaceVariable(mesh=mesh, value=h * n, rank=1)       # a = h * n
# b = FaceVariable(mesh=mesh, value=k_expr.arithmeticFaceValue, rank=0)      # b = k(x), spatially varying
# g = FaceVariable(mesh=mesh, value=h * T_amb, rank=0)   # g = h * T_amb

# # Robin coefficient
# RobinCoeff = mask * D_expr.arithmeticFaceValue * n / (dPf.dot(a) + b)

# # Final equation with Robin boundary condition
# eqn = (TransientTerm(var=T) ==
#        DiffusionTerm(coeff=Gamma, var=T)
#        + source_term_K_s
#        + (RobinCoeff * g).divergence
#        - (ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)), var=T)).divergence)


# ----------------------------------------
# Another try with facegrad - just like example.diffusion.mesh1D
# ----------------------------------------

# wall_faces = dimensions['R_wick'] < y_face
# wall_evap_adia_faces = wall_faces & (x_face < (dimensions['L_e'] + dimensions['L_a']))
# wall_cond_faces = wall_faces & (x_face > (dimensions['L_e'] + dimensions['L_a']))
# wick_faces = (dimensions['R_vc'] < y_face) & (y_face < dimensions['R_wick'])
# vc_faces = (dimensions['R_vc'] >= y_face)
# vc_evap_cond_faces = vc_faces & ((x_face < dimensions['L_e']) | (x_face > (dimensions['L_e'] + dimensions['L_a'])))
# vc_adiabatic_faces = vc_faces & ((x_face > (dimensions['L_e'])) & (x_face < dimensions['L_e'] + dimensions['L_a']))

# preview_face_mask(mesh, wall_faces, title="Wall Faces")
# preview_face_mask(mesh, wall_evap_adia_faces, title="Wall Evaporator and Adiabatic Faces")
# preview_face_mask(mesh, wall_cond_faces, title="Wall Condenser Faces")
# preview_face_mask(mesh, wick_faces, title="Wick Faces")
# preview_face_mask(mesh, vc_faces, title="Vapor Core Faces")
# preview_face_mask(mesh, vc_evap_cond_faces, title="Vapor Core Evaporator and Condenser Faces")
# preview_face_mask(mesh, vc_adiabatic_faces, title="Vapor Core Adiabatic Faces")

# cp = CellVariable(mesh=mesh, value=0., hasOld=True) # TransientTerm expects a CellVariable
# rho = CellVariable(mesh=mesh, value=0., hasOld=True) # TransientTerm expects a CellVariable
# k = FaceVariable(mesh=mesh, value=0., hasOld=True) # DiffusionTerm expects a FaceVariable

# D = FaceVariable(mesh=mesh, value=0.) # DiffusionTerm expects a FaceVariable

# D = 1.0
# cp = 1.0
# rho = 1.0
# k = 1.0

# D.setValue((k_wall_face / (rho_wall_cond_face * c_p_wall_face)), where=wall_cond_faces)
# D.setValue((k_wall_face / (rho_wall_evap_adia_face * c_p_wall_face)), where=wall_evap_adia_faces)
# D.setValue((k_wick_face / (rho_wick_face * c_p_wick_face)), where=wick_faces)
# D.setValue((k_vc_evap_cond_face / (rho_vc_face * c_p_vc_face)), where=vc_evap_cond_faces)
# D.setValue((k_vc_adiabatic_face / (rho_vc_face * c_p_vc_face)), where=vc_adiabatic_faces)

# D = D + ((k_wall_face / (rho_wall_cond_face * c_p_wall_face)) * wall_cond_faces)
# D = D + ((k_wall_face / (rho_wall_evap_adia_face * c_p_wall_face)) * wall_evap_adia_faces)
# D = D + ((k_wick_face / (rho_wick_face * c_p_wick_face)) * wick_faces)
# D = D + ((k_vc_evap_cond_face / (rho_vc_face * c_p_vc_face)) * vc_evap_cond_faces)
# D = D + ((k_vc_adiabatic_face / (rho_vc_face * c_p_vc_face)) * vc_adiabatic_faces)

# cp.setValue(c_p_wall_face, where=wall_faces)
# cp.setValue(c_p_wick_face, where=wick_faces)
# cp.setValue(c_p_vc_face, where=vc_faces)

# cp = (c_p_wall_cell * wall_cells)
# cp = cp + (c_p_wick_cell * wick_cells)
# cp = cp + (c_p_vc_cell * vc_cells)

# rho.setValue(rho_wall_evap_adia_face, where=wall_evap_adia_faces)
# rho.setValue(rho_wall_cond_face, where=wall_cond_faces)
# rho.setValue(rho_wick_face, where=wick_faces)
# rho.setValue(rho_vc_face, where=vc_faces)

# rho = (rho_wall_evap_adia_cell * wall_evap_adia_cells)
# rho = rho + (rho_wall_cond_cell * wall_cond_cells)
# rho = rho + (rho_wick_cell * wick_cells)
# rho = rho + (rho_vc_cell * vc_cells)

# k.setValue(k_wall_face, where=wall_faces)
# k.setValue(k_wick_face, where=wick_faces)
# k.setValue(k_vc_evap_cond_face, where=vc_evap_cond_faces)
# k.setValue(k_vc_adiabatic_face, where=vc_adiabatic_faces)

# k = (k_wall_face * wall_faces)
# k = k + (k_wick_face * wick_faces)
# k = k + (k_vc_evap_cond_face * vc_evap_cond_faces)
# k = k + (k_vc_adiabatic_face * vc_adiabatic_faces)

# k = k + (k_wall_cell * wall_cells)
# k = k + (k_wick_cell * wick_cells)
# k = k + (k_vc_evap_cond_cell * vc_evap_cond_cells)
# k = k + (k_vc_adiabatic_cell * vc_adiabatic_cells)

# D = k / (rho * cp + eps)

# T.faceGrad.constrain(parameters['Q_input_flux']*n, where=faces_evaporator)

# T.constrain(300, where=faces_evaporator)

# q_rad = constants['sigma'] * parameters['emissivity'] * (T.faceValue**4 - parameters['T_amb']**4)

# T.faceGrad.constrain(-q_rad*n, where=faces_condenser)

# T.setValue(all_params["T_amb"])

# ----------------------------------------
# guyer implementation for radiative boundary condition
# ----------------------------------------

# rho_cp_wall_evaporator = rho_wall_evap_adia * c_p_wall
# rho_cp_wall_condenser = rho_wall_cond * c_p_wall

# q_input = FaceVariable(mesh=mesh, value=parameters['Q_input_flux'], rank=1)


# source_term_K_s = ((faces_evaporator * q_input).divergence) / (k_wall + eps)
# sink_term_K_s = ((faces_condenser * q_rad).divergence) / (k_wall + eps)

# D_expr.constrain(0., faces_evaporator)
# # D_expr.constrain(0., faces_condenser)

# eq = TransientTerm(var=T) == (DiffusionTerm(coeff=D_var, var=T)
#                               + source_term_K_s
#                               + sink_term_K_s)

# ----------------------------------------
# Define the PDE
# ----------------------------------------

# eq = TransientTerm(var=T) == DiffusionTerm(coeff=D, var=T)
# eq = TransientTerm(coeff=rho*cp, var=T) == DiffusionTerm(coeff=k, var=T) # + source_term_K_s

# ----------------------------------------
# Dirichlet at condenser
# ----------------------------------------

# T.constrain(parameters['T_amb'], where=faces_condenser)
