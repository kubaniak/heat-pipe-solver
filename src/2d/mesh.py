"""
mesh.py

This module contains functions to generate 2D meshes using FiPy.
"""

from fipy import Grid2D
from fipy import Viewer, CellVariable

from params import get_all_params

def generate_mesh_2d(L_x, L_y, nx, ny):
    """
    Generates a 2D Cartesian mesh using FiPy's Grid2D.

    Parameters:
        L_x (float): Total axial length of the domain [m].
        L_y (float): Total lateral length of the domain [m].
        nx (int): Number of cells in the axial (x) direction.
        ny (int): Number of cells in the lateral (y) direction.

    Returns:
        mesh (Grid2D): A FiPy Grid2D mesh of dimensions L_x by L_y.
    """
    # Compute cell sizes (assuming uniform grid)
    dx = L_x / nx
    dy = L_y / ny

    # Create the 2D grid mesh
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    
    return mesh

import numpy as np
from fipy import Grid2D, CellVariable

from fipy import CellVariable
from fipy.meshes.nonUniformGrid2D import NonUniformGrid2D
import numpy as np

def generate_composite_mesh(mesh_params: dict, dimensions: dict) -> tuple[NonUniformGrid2D, CellVariable]:
    """
    Generate a composite FiPy mesh for a cylindrical heat pipe with variable radial resolution.
    The mesh consists of three regions: vapor core (vc), wick, and wall, each with its own radial resolution.

    Parameters:
        mesh_params (dict): Dictionary containing mesh parameters:
            - "nx_wall" (int): Number of cells in the axial (x) direction for the wall region.
            - "nr_wall" (int): Number of cells in the radial (y) direction for the wall region.
            - "nx_wick" (int): Number of cells in the axial (x) direction for the wick region.
            - "nr_wick" (int): Number of cells in the radial (y) direction for the wick region.
            - "nx_vc" (int): Number of cells in the axial (x) direction for the vapor core region.
            - "nr_vc" (int): Number of cells in the radial (y) direction for the vapor core region.

        dimensions (dict): Dictionary containing geometric dimensions (all in meters):
            - "R_wall" (float): Outer radius of the wall region.
            - "R_wick" (float): Outer radius of the wick region.
            - "R_vc" (float): Outer radius of the vapor core region.
            - "L_e" (float): Length of the evaporator section.
            - "L_a" (float): Length of the adiabatic section.
            - "L_c" (float): Length of the condenser section.

    Returns:
        tuple: A tuple containing:
            - mesh (NonUniformGrid2D): A FiPy NonUniformGrid2D mesh representing the composite geometry.
            - cell_var (CellVariable): A FiPy CellVariable indicating the cell types (0 for vapor core, 1 for wick, 2 for wall).
    """
    # Unpack parameters
    nx = mesh_params["nx_wall"]
    nr = {
        "vc": mesh_params["nr_vc"],
        "wick": mesh_params["nr_wick"],
        "wall": mesh_params["nr_wall"]
    }
    R = {
        "vc": dimensions["R_vc"],
        "wick": dimensions["R_wick"],
        "wall": dimensions["R_wall"]
    }
    lengths = [dimensions["L_e"], dimensions["L_a"], dimensions["L_c"]]
    L_total = sum(lengths)
    dx = L_total / nx

    # Adjust mismatched x-resolution
    for key in ["nx_vc", "nx_wick"]:
        if mesh_params[key] != nx:
            mesh_params[key] = nx

    # Compute dy for each region
    dy = {
        "vc": R["vc"] / nr["vc"],
        "wick": (R["wick"] - R["vc"]) / nr["wick"],
        "wall": (R["wall"] - R["wick"]) / nr["wall"]
    }

    # Build y-vertex coordinates
    vertices_y = [0.0]
    for region in ["vc", "wick", "wall"]:
        for _ in range(nr[region]):
            vertices_y.append(vertices_y[-1] + dy[region])
    dy_array = np.diff(vertices_y)

    # Build x-vertex coordinates (uniform)
    vertices_x = np.linspace(0, L_total, nx + 1)
    dx_array = np.diff(vertices_x)

    # Create non-uniform grid
    mesh = NonUniformGrid2D(dx=dx_array, dy=dy_array, nx=nx, ny=sum(nr.values()))

    # Assign cell types based on cell centers' spatial coordinates
    cell_types = np.zeros(mesh.numberOfCells, dtype=int)
    x, y = mesh.cellCenters  # Get coordinates of cell centers
    
    # Define radial regions
    vc_mask = (y <= R["vc"])
    wick_mask = (y > R["vc"]) & (y <= R["wick"])
    wall_mask = (y > R["wick"])
    
    # Define axial regions
    evap_mask = (x <= dimensions["L_e"]) | (x >= dimensions["L_e"] + dimensions["L_a"])
    adiabatic_mask = (x > dimensions["L_e"]) & (x < dimensions["L_e"] + dimensions["L_a"])
    
    # Assign cell types based on both radial and axial regions
    # Base radial types: vapor core=0, wick=10, wall=20
    # Add axial types: evaporator/condenser=+0, adiabatic=+1
    
    # First set the base cell types for each radial region
    cell_types[vc_mask] = 0     # vapor core base
    cell_types[wick_mask] = 10  # wick base
    cell_types[wall_mask] = 20  # wall base
    
    # Now apply axial conditions for all regions
    # Vapor core regions
    cell_types[vc_mask & evap_mask] = 0     # vapor core in evaporator/condenser sections
    cell_types[vc_mask & adiabatic_mask] = 1  # vapor core in adiabatic section
    
    # Wick regions
    cell_types[wick_mask & evap_mask] = 10    # wick in evaporator/condenser sections
    cell_types[wick_mask & adiabatic_mask] = 11  # wick in adiabatic section
    
    # Wall regions
    cell_types[wall_mask & evap_mask] = 20    # wall in evaporator/condenser sections
    cell_types[wall_mask & adiabatic_mask] = 21  # wall in adiabatic section
    
    # Create a CellVariable for easy visualization
    cell_var = CellVariable(name="Cell Types", mesh=mesh, value=cell_types)

    return mesh, cell_var


# For testing this module independently (optional)
if __name__ == '__main__':
    # Define mesh parameters (example values)
    mesh_params = {
        "nx_wall": 500,
        "nr_wall": 6,
        "nx_wick": 500,
        "nr_wick": 2,
        "nx_vc":   500,
        "nr_vc":   1,
    }

    # Define geometric dimensions (all in meters)
    dimensions = {
        "R_wall": 0.01335,
        "R_wick": 0.0112,
        "R_vc":   0.01075,
        "L_e":    0.502,
        "L_a":    0.188,
        "L_c":    0.292,
    }

    mesh, cell_types = generate_composite_mesh(mesh_params, dimensions)
    print("Composite mesh generated:")
    print(" - Number of x cells:", mesh.numberOfCells // (mesh_params["nr_vc"] + mesh_params["nr_wick"] + mesh_params["nr_wall"]))
    print(" - Total number of cells:", mesh.numberOfCells)

    # Visualize the mesh using the standard preview
    import matplotlib.pyplot as plt
    x, y = mesh.cellCenters
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, marker='o', color='b', s=10)
    plt.title('2D Cartesian Mesh')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # Visualize the different regions using the preview_cell_types function
    from utils import preview_cell_types
    preview_cell_types(mesh, cell_types, title="Heat Pipe Regions Visualization")
