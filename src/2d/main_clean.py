from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, Viewer, FaceVariable, ImplicitSourceTerm
from fipy.tools import numerix as npx
from fipy.meshes import CylindricalGrid2D
from utils import preview_face_mask, preview_cell_mask
from params import get_all_params, get_param_group
from material_properties import get_wick_properties, get_vc_properties, get_steel_properties, get_sodium_properties
from mesh import generate_composite_mesh
from mesh import generate_mesh_2d
from matplotlib import pyplot as plt
import time 
from fipy import parallelComm
import glob # Added import

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
R_total = all_params['R_wall']

# mesh = generate_mesh_2d(L_total, R_total, mesh_params['nx_wall'], 20)
mesh, cell_types = generate_composite_mesh(mesh_params, dimensions)
r_face, z_face = mesh.faceCenters
r_cell, z_cell = mesh.cellCenters
print(f"Number of cells: {mesh.numberOfCells}")

T = CellVariable(name="Temperature", mesh=mesh, value=all_params["T_amb"], hasOld=False)

T_amb = parameters['T_amb'] # Used as reference temperature (e.g., 290K)
h = 5.0

# region simple properties
# constants
# rho_0 = 7200.0
# rho_0 = CellVariable(mesh=mesh, value=rho_0) # For spatial dependence, we have to define this explicitly
# cp_0 = 440.5
# cp_0 = CellVariable(mesh=mesh, value=cp_0) # For spatial dependence, we have to define this explicitly
# k_0 = 55.0
# k_0 = FaceVariable(mesh=mesh, value=k_0) # For spatial dependence, we have to define this explicitly
# Simple temperature-dependent properties
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
# endregion

# region property masks
vc_cells = (r_cell < dimensions['R_vc'])
vc_evap_cond_cells = vc_cells & ((z_cell < dimensions['L_e']) | (z_cell > (dimensions['L_e'] + dimensions['L_a'])))
vc_adiabatic_cells = vc_cells & ((z_cell > dimensions['L_e']) & (z_cell < dimensions['L_e'] + dimensions['L_a']))
wick_cells = ((dimensions['R_vc'] < r_cell) & (r_cell < dimensions['R_wick']))
wick_evap_adia_cells = wick_cells & ((z_cell < dimensions['L_e'] + dimensions['L_a']))
wick_cond_cells = wick_cells & (z_cell > (dimensions['L_e'] + dimensions['L_a']))
wall_cells = (r_cell > dimensions['R_wick'])
wall_evap_adia_cells = wall_cells & ((z_cell < dimensions['L_e'] + dimensions['L_a']))
wall_cond_cells = wall_cells & (z_cell > (dimensions['L_e'] + dimensions['L_a']))

# preview_cell_mask(mesh, vc_evap_cond_cells, title="Vapor Core Evaporator and Condenser Region")
# preview_cell_mask(mesh, vc_adiabatic_cells, title="Vapor Core Adiabatic Region")
# preview_cell_mask(mesh, vc_cells, title="Vapor Core Region")
# preview_cell_mask(mesh, wick_cells, title="Wick Region")
# preview_cell_mask(mesh, wick_evap_adia_cells, title="Wick Evaporator and Adiabatic Region")
# preview_cell_mask(mesh, wick_cond_cells, title="Wick Condenser Region")
# preview_cell_mask(mesh, wall_cells, title="Wall Region")
# preview_cell_mask(mesh, wall_evap_adia_cells, title="Wall Evaporator and Adiabatic Region")
# preview_cell_mask(mesh, wall_cond_cells, title="Wall Condenser Region")

# preview_cell_mask(mesh, vc_evap_cond_cells | vc_adiabatic_cells | wick_cells | wall_cells, title="All Cells") 

vc_faces = (r_face <= dimensions['R_vc'])
vc_evap_cond_faces = vc_faces & ((z_face < dimensions['L_e']) | (z_face > (dimensions['L_e'] + dimensions['L_a'])))
vc_adiabatic_faces = vc_faces & ((z_face > dimensions['L_e']) & (z_face < dimensions['L_e'] + dimensions['L_a']))
wick_faces = ((dimensions['R_vc'] < r_face) & (r_face < dimensions['R_wick']))
wall_faces = (r_face > dimensions['R_wick'])

# preview_face_mask(mesh, wall_faces, title="Wall Faces")
# preview_face_mask(mesh, wall_evap_adia_faces, title="Wall Evaporator and Adiabatic Faces")
# preview_face_mask(mesh, wall_cond_faces, title="Wall Condenser Faces")
# preview_face_mask(mesh, wick_faces, title="Wick Faces")
# preview_face_mask(mesh, vc_faces, title="Vapor Core Faces")
# preview_face_mask(mesh, vc_evap_cond_faces, title="Vapor Core Evaporator and Condenser Faces")
# preview_face_mask(mesh, vc_adiabatic_faces, title="Vapor Core Adiabatic Faces")

# preview_face_mask(mesh, vc_evap_cond_faces | vc_adiabatic_faces | wick_faces | wall_faces, title="All Faces")
# endregion

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
    + steel_properties['density_evap_adia'](T) * (wall_evap_adia_cells) \
    + steel_properties['density_cond'](T) * (wall_cond_cells)

t_end = 101.0
dt = 0.1

# Define boundary condition masks
faces_evaporator = (mesh.facesRight & ((z_face < dimensions['L_input_right']) & (z_face > dimensions['L_input_left'])))
faces_condenser = (mesh.facesRight & (z_face > (dimensions['L_e'] + dimensions['L_a'])) & (z_face < L_total))

# preview_face_mask(mesh, faces_evaporator, title="Evaporator Faces")
# preview_face_mask(mesh, faces_condenser, title="Condenser Faces")

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
eq = (TransientTerm(coeff=cp*rho, var=T) == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff * g).divergence
       - ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)).divergence, var=T)
       + (faces_evaporator * (q/k)).divergence)

T.setValue(T_amb)  # Set initial temperature

# viewer = Viewer(vars=T, title="Temperature Distribution")
# viewer.plot()

# measure_times = [1038, 1998, 2958]
measure_times = [] 
measure_times.extend(range(5, int(t_end), 5))
# measure_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # For testing purposes
print(f"Measuring at: {measure_times} seconds")

# region plotting
# Find topmost wall cells: any face of the cell is a top face
cellFaceIDs = npx.array(mesh.cellFaceIDs)  # shape: (nFacesPerCell, nCells)
facesTop = npx.array(mesh.facesRight)        # shape: (nFaces,)

# For each cell, check if any of its faces are top faces
top_wall_mask = wall_cells & npx.any(facesTop[cellFaceIDs], axis=0)

# preview_cell_mask(mesh, top_wall_mask, title="Top Wall Cells")

npx.set_printoptions(linewidth=200)

# Initialize lists for storing profiles and their corresponding times
profiles = []
actual_profile_times = []
next_measure_time_idx = 0

# Initialize lists for storing density profiles
rho_top_wall_profiles = []
rho_wick_profiles = []
rho_vc_profiles = []

# Initialize lists for storing specific heat profiles
cp_top_wall_profiles = []
cp_wick_profiles = []
cp_vc_profiles = []

# Initialize lists for storing thermal conductivity profiles
k_top_wall_profiles = []
k_wick_profiles = []
k_vc_profiles = []
# endregion

start_time = time.time() # Record start time
print(f"Simulating for {t_end} seconds...")

# region Solver WITHOUT temperature-dependent properties (DON'T FORGET hasOld=False!)
for t in npx.arange(0, t_end, dt):
    # eq.solve(var=T, dt=dt)
    res = eq.sweep(var=T, dt=dt)
    # Capture profile if current time t is at or just past a scheduled measure_time
    if next_measure_time_idx < len(measure_times) and t >= measure_times[next_measure_time_idx]:
        current_time_step = t # Use the actual time t from the loop for consistency
        if not actual_profile_times or actual_profile_times[-1] < current_time_step: # Ensure we only add once per measure point
            print(f"Time {current_time_step:.2f}, Residual: {res}")
            profiles.append(T.value[top_wall_mask].copy()) # Temperature
            actual_profile_times.append(current_time_step)

            # Capture density profiles
            current_rho_values = rho.value # Get current rho values once
            rho_top_wall_profiles.append(current_rho_values[top_wall_mask].copy())
            rho_vc_profiles.append(current_rho_values[vc_cells].copy())

            # Handle wick cells density - average for each unique x-coordinate
            current_rho_wick_all_cells = current_rho_values[wick_cells].copy()
            current_x_wick_all_cells = z_cell[wick_cells]
            
            unique_x_wick = npx.unique(current_x_wick_all_cells) # Sorted unique x-coordinates
            averaged_rho_wick_profile = npx.array([
                npx.mean(current_rho_wick_all_cells[current_x_wick_all_cells == ux]) for ux in unique_x_wick
            ])
            rho_wick_profiles.append(averaged_rho_wick_profile)

            # Capture specific heat profiles
            current_cp_values = cp.value # Get current cp values once
            cp_top_wall_profiles.append(current_cp_values[top_wall_mask].copy())
            cp_vc_profiles.append(current_cp_values[vc_cells].copy())

            # Handle wick cells specific heat - average for each unique x-coordinate
            current_cp_wick_all_cells = current_cp_values[wick_cells].copy()
            # current_x_wick_all_cells is the same as for rho, unique_x_wick is also the same
            averaged_cp_wick_profile = npx.array([
                npx.mean(current_cp_wick_all_cells[current_x_wick_all_cells == ux]) for ux in unique_x_wick
            ])
            cp_wick_profiles.append(averaged_cp_wick_profile)

            # Capture thermal conductivity profiles (k is FaceVariable, average to cell centers first)
            current_k_face_values = k.value
            # cellFaceIDs was defined earlier as npx.array(mesh.cellFaceIDs)
            # Average k values of faces surrounding each cell to get a cell-centered k
            cell_avg_k = npx.mean(current_k_face_values[cellFaceIDs], axis=0)
            
            k_top_wall_profiles.append(cell_avg_k[top_wall_mask].copy())
            k_vc_profiles.append(cell_avg_k[vc_cells].copy())

            # Handle wick cells thermal conductivity - average for each unique x-coordinate
            current_k_wick_all_cells = cell_avg_k[wick_cells].copy()
            # current_x_wick_all_cells and unique_x_wick are already defined and can be reused from density/cp calculation
            averaged_k_wick_profile = npx.array([
                npx.mean(current_k_wick_all_cells[current_x_wick_all_cells == ux]) for ux in unique_x_wick
            ])
            k_wick_profiles.append(averaged_k_wick_profile)
            
        next_measure_time_idx += 1
        # Ensure we advance past all measure_times that are less than or equal to current t
        while next_measure_time_idx < len(measure_times) and measure_times[next_measure_time_idx] <= t:
            next_measure_time_idx +=1

    # if t % 5 == 0:
    #     if __name__ == "__main__":
    #         viewer.plot()
# endregion

# # region Solver WITH temperature-dependent properties (DON'T FORGET hasOld=True!)
# sweeps = 5
# for t in npx.arange(0, t_end, dt):
#     T.updateOld()
#     for sweep in range(sweeps):
#         res = eq.sweep(var=T, dt=dt)
#         # print(f"Time {t:.2f}, Sweep {sweep}, Residual: {res}")
#     # if t % 5 == 0:
#     #     if __name__ == "__main__":
#     #         viewer.plot()

#     if next_measure_time_idx < len(measure_times) and t >= measure_times[next_measure_time_idx]:
#         current_time_step = t # Use the actual time t from the loop for consistency
#         if not actual_profile_times or actual_profile_times[-1] < current_time_step: # Ensure we only add once per measure point
#             print(f"Time {current_time_step:.2f}, Residual: {res}")
#             profiles.append(T.value[top_wall_mask].copy()) # Temperature
#             actual_profile_times.append(current_time_step)

#             # Capture density profiles
#             current_rho_values = rho.value # Get current rho values once
#             rho_top_wall_profiles.append(current_rho_values[top_wall_mask].copy())
#             rho_vc_profiles.append(current_rho_values[vc_cells].copy())

#             # Handle wick cells density - average for each unique x-coordinate
#             current_rho_wick_all_cells = current_rho_values[wick_cells].copy()
#             current_x_wick_all_cells = z_cell[wick_cells]
            
#             unique_x_wick = npx.unique(current_x_wick_all_cells) # Sorted unique x-coordinates
#             averaged_rho_wick_profile = npx.array([
#                 npx.mean(current_rho_wick_all_cells[current_x_wick_all_cells == ux]) for ux in unique_x_wick
#             ])
#             rho_wick_profiles.append(averaged_rho_wick_profile)

#             # Capture specific heat profiles
#             current_cp_values = cp.value # Get current cp values once
#             cp_top_wall_profiles.append(current_cp_values[top_wall_mask].copy())
#             cp_vc_profiles.append(current_cp_values[vc_cells].copy())

#             # Handle wick cells specific heat - average for each unique x-coordinate
#             current_cp_wick_all_cells = current_cp_values[wick_cells].copy()
#             # current_x_wick_all_cells is the same as for rho, unique_x_wick is also the same
#             averaged_cp_wick_profile = npx.array([
#                 npx.mean(current_cp_wick_all_cells[current_x_wick_all_cells == ux]) for ux in unique_x_wick
#             ])
#             cp_wick_profiles.append(averaged_cp_wick_profile)

#             # Capture thermal conductivity profiles (k is FaceVariable, average to cell centers first)
#             current_k_face_values = k.value
#             # cellFaceIDs was defined earlier as npx.array(mesh.cellFaceIDs)
#             # Average k values of faces surrounding each cell to get a cell-centered k
#             cell_avg_k = npx.mean(current_k_face_values[cellFaceIDs], axis=0)
            
#             k_top_wall_profiles.append(cell_avg_k[top_wall_mask].copy())
#             k_vc_profiles.append(cell_avg_k[vc_cells].copy())

#             # Handle wick cells thermal conductivity - average for each unique x-coordinate
#             current_k_wick_all_cells = cell_avg_k[wick_cells].copy()
#             # current_x_wick_all_cells and unique_x_wick are already defined and can be reused from density/cp calculation
#             averaged_k_wick_profile = npx.array([
#                 npx.mean(current_k_wick_all_cells[current_x_wick_all_cells == ux]) for ux in unique_x_wick
#             ])
#             k_wick_profiles.append(averaged_k_wick_profile)
            
#         next_measure_time_idx += 1
#         # Ensure we advance past all measure_times that are less than or equal to current t
#         while next_measure_time_idx < len(measure_times) and measure_times[next_measure_time_idx] <= t:
#             next_measure_time_idx +=1
# # endregion

end_time = time.time() # Record end time
elapsed_time = end_time - start_time
print(f"Simulated time: {t_end} seconds")
print(f"Actual execution time: {elapsed_time:.2f} seconds")
print("%d cells on processor %d of %d" % (mesh.numberOfCells, parallelComm.procID, parallelComm.Nproc))

# region plotting
# Import the plotting function from your other script
# Make sure plot_riccardo_data.py is in a location where Python can find it (e.g., same directory or in PYTHONPATH)
# For this example, assuming it's in the same directory or src/2d/
import sys
import os
# Add the directory containing plot_riccardo_data to sys.path if it's not already discoverable
# This assumes plot_riccardo_data.py is in the same directory as main_clean.py (src/2d)
module_path = os.path.dirname(__file__) 
if module_path not in sys.path:
    sys.path.append(module_path)
try:
    from plot_riccardo_data import plot_property_over_time as plot_riccardo_property
    can_plot_riccardo = True
except ImportError:
    print("Warning: Could not import 'plot_riccardo_data'. Riccardo data overlay will be disabled.")
    can_plot_riccardo = False

# Toggle for Riccardo data comparison
PLOT_RICCARDO_COMPARISON = True # Set to False to disable Riccardo data overlay

# Get x-coordinates for top wall cells
x_top_wall = z_cell[top_wall_mask]

# Sort by x for plotting
sort_idx = x_top_wall.argsort()
x_top_wall_sorted = x_top_wall[sort_idx]

import pandas as pd

# Define paths to experimental data files and their corresponding times in seconds
experimental_data_files = {
    1038: "Faghri_Data/Faghiri_17_3.csv",  # 17.3 minutes
    1998: "Faghri_Data/Faghiri_33_3.csv",  # 33.3 minutes
    2958: "Faghri_Data/Faghiri_49_3.csv"   # 49.3 minutes
}
# Tolerance for matching simulation time with experimental time keys
# dt is defined earlier in the script (e.g., dt = 0.03)
time_tolerance = dt 

plt.figure(figsize=(10, 6))
for i, (T_profile, t_plot) in enumerate(zip(profiles, actual_profile_times)): # Use actual_profile_times
    plt.plot(x_top_wall_sorted, T_profile[sort_idx], label=f"Sim. t = {t_plot:.2f} s ({t_plot/60:.1f} min)")
    # Additions for experimental data plotting
    for exp_time_s, file_path in experimental_data_files.items():
        if abs(t_plot - exp_time_s) < time_tolerance:
            try:
                # Assuming the script is run from the workspace root directory
                # where 'Faghri_Data' is a subdirectory.
                exp_df = pd.read_csv(file_path)
                
                temp_to_plot = exp_df['y'] 
                
                plt.scatter(exp_df['x'], temp_to_plot, marker='o', s=40, edgecolor='black', facecolor='none',
                            label=f"Exp. ({exp_time_s/60:.1f} min)")
            except FileNotFoundError:
                print(f"Warning: Experimental data file not found: {file_path}")
            except KeyError:
                # This could happen if CSV columns are not named 'x' and 'y'
                print(f"Warning: CSV file {file_path} might be missing 'x' or 'y' columns.")
            except Exception as e:
                print(f"Warning: Could not plot experimental data from {file_path}. Error: {e}")
            # Break after attempting to plot for the first matching experimental time
            # This assumes one experimental dataset per simulation snapshot.
            break 

plt.xlabel("x [m]")
plt.ylabel("Temperature [K]")
plt.title("Axial Temperature Profile in Topmost Wall Cells at Different Times")
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/top_wall_axial_profiles.png", dpi=300)
plt.show()

# --- Generic plotting function for simulation data ---
def plot_simulation_data(x_coords, sim_profiles, sim_times, title, ylabel, output_filename, 
                         sort_indices=None, # Added for cases where x_coords might not be pre-sorted with profiles
                         riccardo_region_code=None, riccardo_property_column=None, riccardo_base_path=None):
    plt.figure(figsize=(10, 6))
    ax = plt.gca() # Get current axes for potential overlay

    # Plot simulation data
    for i, (profile, t_plot) in enumerate(zip(sim_profiles, sim_times)):
        if sort_indices is not None:
            ax.plot(x_coords, profile[sort_indices], label=f"Sim. t = {t_plot:.2f} s")
        else:
            ax.plot(x_coords, profile, label=f"Sim. t = {t_plot:.2f} s") # Assumes x_coords and profile are already aligned/sorted

    # Overlay Riccardo data if enabled and applicable
    if PLOT_RICCARDO_COMPARISON and can_plot_riccardo and riccardo_region_code and riccardo_property_column and riccardo_base_path:
        riccardo_file_pattern = os.path.join(riccardo_base_path, f'{riccardo_region_code}_properties_Faghri_ax1000_TS_bra_StepFct05_properties_h5_*.000_s.csv')
        riccardo_files = glob.glob(riccardo_file_pattern)
        if riccardo_files:
            print(f"Overlaying Riccardo data for {riccardo_region_code} - {riccardo_property_column} on {title}")
            plot_riccardo_property(
                file_paths=riccardo_files,
                property_column_name=riccardo_property_column,
                y_label=ylabel, # y_label is the same
                title=title, # Title is managed by the main plot
                output_filename="", # Not saving separately
                ax=ax, # Pass the current axes
                label_prefix="Riccardo ",
                plot_kwargs={'linestyle': '--'} # Dashed lines for Riccardo data
            )
        else:
            print(f"Warning: No Riccardo files found for {riccardo_region_code} at {riccardo_base_path}")

    ax.set_xlabel("x [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

# Define base path for Riccardo data (relative to workspace root)
riccardo_data_base_path = os.path.join(os.getcwd(), "Properties_riccardo", "Properties")

# Plotting for Density in Top Wall Cells
plot_simulation_data(
    x_top_wall_sorted, rho_top_wall_profiles, actual_profile_times,
    "Axial Density Profile in Topmost Wall Cells at Different Times", "Density [kg/m^3]",
    "plots/top_wall_density_profiles.png",
    sort_indices=sort_idx, # Pass the sort_idx for top wall
    riccardo_region_code="Wall", riccardo_property_column='Density (kg/m^3)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Density in Vapor Core Cells
x_vc_all = z_cell[vc_cells]
_sort_idx_vc = x_vc_all.argsort() # Renamed to avoid conflict if main_clean uses sort_idx_vc later
x_vc_sorted_for_plot = x_vc_all[_sort_idx_vc]

plot_simulation_data(
    x_vc_sorted_for_plot, rho_vc_profiles, actual_profile_times,
    "Axial Density Profile in Vapor Core Cells at Different Times", "Density [kg/m^3]",
    "plots/vc_density_profiles.png",
    sort_indices=_sort_idx_vc, # Pass the sort_idx for VC
    riccardo_region_code="VC", riccardo_property_column='Density (kg/m^3)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Density in Wick Cells (Averaged)
x_wick_plot_coords_for_plot = npx.unique(z_cell[wick_cells]) # Define before use

plot_simulation_data(
    x_wick_plot_coords_for_plot, rho_wick_profiles, actual_profile_times,
    "Axial Density Profile in Wick Cells at Different Times", "Average Density [kg/m^3]",
    "plots/wick_density_profiles.png",
    # No sort_indices needed here as rho_wick_profiles are already averaged and aligned with unique x_wick_plot_coords
    riccardo_region_code="Wick", riccardo_property_column='Density (kg/m^3)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Specific Heat in Top Wall Cells
plot_simulation_data(
    x_top_wall_sorted, cp_top_wall_profiles, actual_profile_times,
    "Axial Specific Heat Profile in Topmost Wall Cells at Different Times", "Specific Heat [J/kg*K]",
    "plots/top_wall_cp_profiles.png",
    sort_indices=sort_idx, # Pass the sort_idx for top wall
    riccardo_region_code="Wall", riccardo_property_column='Specific Heat (J/kg-K)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Specific Heat in Vapor Core Cells
# x_vc_sorted_for_plot and _sort_idx_vc are already defined from density plots
plot_simulation_data(
    x_vc_sorted_for_plot, cp_vc_profiles, actual_profile_times,
    "Axial Specific Heat Profile in Vapor Core Cells at Different Times", "Specific Heat [J/kg*K]",
    "plots/vc_cp_profiles.png",
    sort_indices=_sort_idx_vc, # Pass the sort_idx for VC
    riccardo_region_code="VC", riccardo_property_column='Specific Heat (J/kg-K)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Specific Heat in Wick Cells (Averaged)
# x_wick_plot_coords_for_plot is already defined from density plots
plot_simulation_data(
    x_wick_plot_coords_for_plot, cp_wick_profiles, actual_profile_times,
    "Axial Specific Heat Profile in Wick Cells at Different Times", "Average Specific Heat [J/kg*K]",
    "plots/wick_cp_profiles.png",
    riccardo_region_code="Wick", riccardo_property_column='Specific Heat (J/kg-K)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Thermal Conductivity in Top Wall Cells
# x_top_wall_sorted and sort_idx are already defined
plot_simulation_data(
    x_top_wall_sorted, k_top_wall_profiles, actual_profile_times,
    "Axial Thermal Conductivity Profile in Topmost Wall Cells at Different Times", "Thermal Conductivity [W/m*K]",
    "plots/top_wall_k_profiles.png",
    sort_indices=sort_idx, # Pass the sort_idx for top wall
    riccardo_region_code="Wall", riccardo_property_column='Thermal Conductivity (W/m-K)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Thermal Conductivity in Vapor Core Cells
# x_vc_sorted_for_plot and _sort_idx_vc are already defined
plot_simulation_data(
    x_vc_sorted_for_plot, k_vc_profiles, actual_profile_times,
    "Axial Thermal Conductivity Profile in Vapor Core Cells at Different Times", "Thermal Conductivity [W/m*K]",
    "plots/vc_k_profiles.png",
    sort_indices=_sort_idx_vc, # Pass the sort_idx for VC
    riccardo_region_code="VC", riccardo_property_column='Thermal Conductivity (W/m-K)', riccardo_base_path=riccardo_data_base_path
)

# Plotting for Thermal Conductivity in Wick Cells (Averaged)
# x_wick_plot_coords_for_plot is already defined
plot_simulation_data(
    x_wick_plot_coords_for_plot, k_wick_profiles, actual_profile_times,
    "Axial Thermal Conductivity Profile in Wick Cells at Different Times", "Average Thermal Conductivity [W/m*K]",
    "plots/wick_k_profiles.png",
    riccardo_region_code="Wick", riccardo_property_column='Thermal Conductivity (W/m-K)', riccardo_base_path=riccardo_data_base_path
)
# endregion

input("Press Enter to continue...")