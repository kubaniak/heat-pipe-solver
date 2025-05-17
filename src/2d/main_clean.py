from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, Viewer, FaceVariable, ImplicitSourceTerm
from fipy.tools import numerix as npx
from fipy.meshes import CylindricalGrid2D
from utils import preview_face_mask, preview_cell_mask
from params import get_all_params, get_param_group
from material_properties import get_wick_properties, get_vc_properties, get_steel_properties, get_sodium_properties
from mesh import generate_composite_mesh
from mesh import generate_mesh_2d
from tqdm import tqdm
from matplotlib import pyplot as plt

all_params = get_all_params()
mesh_params = get_param_group('mesh')
dimensions = get_param_group('dimensions')
parameters = get_param_group('parameters')
constants = get_param_group('constants')

vc_properties = get_vc_properties()
steel_properties = get_steel_properties()
wick_properties = get_wick_properties()
sodium_properties = get_sodium_properties()

L_total = all_params['L_e'] + all_params['L_a'] + all_params['L_c']
R_total = all_params['R_wick'] + all_params['R_wall'] + all_params['R_vc']

# mesh = generate_mesh_2d(L_total, R_total, mesh_params['nx_wall'], 20)
mesh, cell_types = generate_composite_mesh(mesh_params, dimensions)
x_face, y_face = mesh.faceCenters
x_cell, y_cell = mesh.cellCenters

T = CellVariable(name="Temperature", mesh=mesh, value=all_params["T_amb"], hasOld=False)


# preview_face_mask(mesh, faces_evaporator, title="Evaporator Faces")
# preview_face_mask(mesh, faces_condenser, title="Condenser Faces")

# constants
# rho_0 = 7200.0
# rho_0 = CellVariable(mesh=mesh, value=rho_0) # For spatial dependence, we have to define this explicitly
# cp_0 = 440.5
# cp_0 = CellVariable(mesh=mesh, value=cp_0) # For spatial dependence, we have to define this explicitly
# k_0 = 55.0
# k_0 = FaceVariable(mesh=mesh, value=k_0) # For spatial dependence, we have to define this explicitly
T_amb = parameters['T_amb'] # Used as reference temperature (e.g., 290K)
h = 750.0

"""# Simple temperature-dependent properties
# cp_T = cp_0 * (1 + 0.005 * (T - T_amb)) # cp is defined at cell centers, so we can use T directly
# k_T = k_0 * (1 + 0.005 * (T.faceValue - T_amb)) # T.faceValue is used because k is the diffusion coefficient, defined between faces
# rho_T = rho_0 * (1 + 0.005 * (T - T_amb)) # rho is defined at cell centers, so we can use T directly

# Simple spatially-dependent properties
# cp_x = cp_0 * 1e-3 * (x_cell >= dimensions['L_input_right'] - 0.02) + cp_0 * 1e3 * (x_cell < dimensions['L_input_right'] - 0.02)
# k_x = k_0 * 1e1 * (x_face >= dimensions['L_input_right'] - 0.02) + k_0 * 1e-1 * (x_face < dimensions['L_input_right'] - 0.02)
# rho_x = rho_0 * 1e-3 * (x_cell >= dimensions['L_input_right'] - 0.02) + rho_0 * 1e3 * (x_cell < dimensions['L_input_right'] - 0.02)

# Simple spatially-dependent and temperature-dependent properties
# cp_Tx = cp_0 * (1 + 0.005 * (T - T_amb)) * 1e-3 * (x_cell >= dimensions['L_input_right'] - 0.02) + cp_0 * (1 + 0.005 * (T - T_amb)) * 1e3 * (x_cell < dimensions['L_input_right'] - 0.02)
# k_Tx = k_0 * (1 + 0.005 * (T.faceValue - T_amb)) * 1e1 * (x_face >= dimensions['L_input_right'] - 0.02) + k_0 * (1 + 0.005 * (T.faceValue - T_amb)) * 1e-1 * (x_face < dimensions['L_input_right'] - 0.02)
# rho_Tx = rho_0 * (1 + 0.005 * (T - T_amb)) * 1e-3 * (x_cell >= dimensions['L_input_right'] - 0.02) + rho_0 * (1 + 0.005 * (T - T_amb)) * 1e3 * (x_cell < dimensions['L_input_right'] - 0.02)
"""

# Property masks
vc_evap_cond_cells = (cell_types == 0)
vc_adiabatic_cells = (cell_types == 1)
vc_cells = vc_evap_cond_cells | vc_adiabatic_cells
wick_cells = (cell_types == 10) | (cell_types == 11)
wall_cells = (cell_types == 20) | (cell_types == 21)
wall_cond_cells = wall_cells & (x_cell > dimensions['L_e'] + dimensions['L_a'])
wall_evap_adia_cells = wall_cells & (x_cell < dimensions['L_e'] + dimensions['L_a']) # Wall in evaporator and adiabatic regions

# preview_cell_mask(mesh, vc_evap_cond_cells, title="Vapor Core Evaporator and Condenser Region")
# preview_cell_mask(mesh, vc_adiabatic_cells, title="Vapor Core Adiabatic Region")
# preview_cell_mask(mesh, wick_cells, title="Wick Region")
# preview_cell_mask(mesh, wall_cells, title="Wall Region")
# preview_cell_mask(mesh, wall_cond_cells, title="Wall Condenser Region")
# preview_cell_mask(mesh, wall_evap_adia_cells, title="Wall Evaporator and Adiabatic Region")

# preview_cell_mask(mesh, vc_evap_cond_cells | vc_adiabatic_cells | wick_cells | wall_cells, title="All Cells") 

vc_faces = (dimensions['R_vc'] >= y_face)
vc_evap_cond_faces = vc_faces & ((x_face < dimensions['L_e']) | (x_face > (dimensions['L_e'] + dimensions['L_a'])))
vc_adiabatic_faces = vc_faces & ((x_face > (dimensions['L_e'])) & (x_face < dimensions['L_e'] + dimensions['L_a']))
wick_faces = (dimensions['R_vc'] < y_face) & (y_face < dimensions['R_wick'])
wall_faces = dimensions['R_wick'] < y_face
wall_evap_adia_faces = wall_faces & (x_face < (dimensions['L_e'] + dimensions['L_a']))
wall_cond_faces = wall_faces & (x_face > (dimensions['L_e'] + dimensions['L_a']))

# preview_face_mask(mesh, wall_faces, title="Wall Faces")
# preview_face_mask(mesh, wall_evap_adia_faces, title="Wall Evaporator and Adiabatic Faces")
# preview_face_mask(mesh, wall_cond_faces, title="Wall Condenser Faces")
# preview_face_mask(mesh, wick_faces, title="Wick Faces")
# preview_face_mask(mesh, vc_faces, title="Vapor Core Faces")
# preview_face_mask(mesh, vc_evap_cond_faces, title="Vapor Core Evaporator and Condenser Faces")
# preview_face_mask(mesh, vc_adiabatic_faces, title="Vapor Core Adiabatic Faces")

# preview_face_mask(mesh, vc_evap_cond_faces | vc_adiabatic_faces | wick_faces | wall_faces, title="All Faces")

# Real temperature-dependent properties
k = vc_properties['thermal_conductivity'](T.faceValue, mesh, 'evap_cond', sodium_properties, dimensions, parameters, constants) * (vc_evap_cond_faces) \
    + vc_properties['thermal_conductivity'](T.faceValue, mesh, 'adiabatic', sodium_properties, dimensions, parameters, constants) * (vc_adiabatic_faces) \
    + wick_properties['thermal_conductivity'](T.faceValue, sodium_properties, steel_properties, parameters) * (wick_faces) \
    + steel_properties['thermal_conductivity'](T.faceValue) * (wall_faces)

cp = vc_properties['specific_heat'](T) * (vc_evap_cond_cells) \
    + vc_properties['specific_heat'](T) * (vc_adiabatic_cells) \
    + wick_properties['specific_heat'](T, sodium_properties, steel_properties, parameters) * (wick_cells) \
    + steel_properties['specific_heat'](T) * (wall_cells)

rho = vc_properties['density'](T) * (vc_evap_cond_cells) \
    + vc_properties['density'](T) * (vc_adiabatic_cells) \
    + wick_properties['density'](T, sodium_properties, steel_properties, parameters) * (wick_cells) \
    + steel_properties['density'](T) * (wall_cells)


# Define boundary condition masks
faces_evaporator = (mesh.facesTop & ((x_face < dimensions['L_input_right']) & (x_face > dimensions['L_input_left'])))
faces_condenser = (mesh.facesTop & (x_face > (dimensions['L_e'] + dimensions['L_a'])) & (x_face < L_total))

# Constant influx
q = parameters['Q_input_flux']

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
eq = (TransientTerm(coeff=rho*cp, var=T) == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff * g).divergence
       - ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)).divergence, var=T)
       + (faces_evaporator * (q/k)).divergence)

T.setValue(T_amb)  # Set initial temperature

dt = 0.02
t_end = 600.0

# viewer = Viewer(vars=T, title="Temperature Distribution")
# viewer.plot()

measure_times = [1, 30, 60, 120, 300, 400, 500, 550] # seconds

# Find topmost wall cells: any face of the cell is a top face
cellFaceIDs = npx.array(mesh.cellFaceIDs)  # shape: (nFacesPerCell, nCells)
facesTop = npx.array(mesh.facesTop)        # shape: (nFaces,)

# For each cell, check if any of its faces are top faces
top_wall_mask = wall_cells & npx.any(facesTop[cellFaceIDs], axis=0)

# preview_cell_mask(mesh, top_wall_mask, title="Top Wall Cells")

npx.set_printoptions(linewidth=200)

# Initialize lists for storing profiles and their corresponding times
profiles = []
actual_profile_times = []
next_measure_time_idx = 0

# Solver WITHOUT temperature-dependent properties (DON'T FORGET hasOld=False!)
for t in npx.arange(0, t_end, dt):
    # print(f"rho_x: {cp_x}")
    # print(f"cp_x: {k_Tx}")
    # print(f"k_x: {rho_x}")
    # eq.solve(var=T, dt=dt)
    residual = eq.sweep(var=T, dt=dt)
    print(f"Time {t:.2f}, Residual: {residual}")

    # Capture profile if current time t is at or just past a scheduled measure_time
    if next_measure_time_idx < len(measure_times) and t >= measure_times[next_measure_time_idx]:
        profiles.append(T.value[top_wall_mask].copy())
        actual_profile_times.append(t)
        next_measure_time_idx += 1
        
    # if t % 5 == 0:
    #     if __name__ == "__main__":
    #         viewer.plot()

"""# Solver WITH temperature-dependent properties (DON'T FORGET hasOld=True!)
# sweeps = 5
# timestep = 0
# for t in range(int(t_end/dt)):
#     T.updateOld()
#     for sweep in range(sweeps):
#         res = eq.sweep(var=T, dt=dt)
#         print(f"Iteration {t}, Sweep {sweep}, Residual: {res}")
#     if __name__ == "__main__":
#         viewer.plot()"""

# Get x-coordinates for top wall cells
x_top_wall = x_cell[top_wall_mask]

# Sort by x for plotting
sort_idx = x_top_wall.argsort()
x_top_wall_sorted = x_top_wall[sort_idx]

plt.figure(figsize=(10, 6))
for i, (T_profile, t_plot) in enumerate(zip(profiles, actual_profile_times)): # Use actual_profile_times
    plt.plot(x_top_wall_sorted, T_profile[sort_idx], label=f"t = {t_plot:.2f} s")
plt.xlabel("x [m]")
plt.ylabel("Temperature [K]")
plt.title("Axial Temperature Profile in Topmost Wall Cells at Different Times")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/top_wall_axial_profiles.png", dpi=300)
plt.show()

from fipy import input
input("Press Enter to continue...")