"""
mesh.py

This module contains functions to generate 2D meshes using FiPy.
For now, we generate a simple Cartesian grid using Grid2D.
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

def generate_composite_mesh(mesh_params, dimensions):
    """
    Generate a composite mesh for a heat pipe with different cell densities in different regions.
    
    The mesh has three radial sections:
    - Wall (outer layer): finer mesh
    - Wick (middle layer): medium mesh density
    - Vapor core (inner core): coarse mesh
    
    Parameters:
        mesh_params (dict): Dictionary containing mesh parameters:
            - nx_wall, nr_wall: Number of cells in x and r direction for the wall
            - nx_wick, nr_wick: Number of cells in x and r direction for the wick
            - nx_vc, nr_vc: Number of cells in x and r direction for the vapor core
        
        dimensions (dict): Dictionary containing geometric dimensions in meters:
            - R_wall: Outer radius of the wall
            - R_wick: Outer radius of the wick (inner radius of wall)
            - R_vc: Radius of the vapor core (inner radius of wick)
            - L_e, L_a, L_c: Lengths of evaporator, adiabatic section, and condenser
    
    Returns:
        mesh (Grid2D): A FiPy Grid2D mesh representing the heat pipe
        cell_types (CellVariable): Cell variable indicating the region type (0: vapor core, 1: wick, 2: wall)
    """
    # Calculate total length
    L_total = dimensions["L_e"] + dimensions["L_a"] + dimensions["L_c"]
    
    # Extract mesh parameters
    nx_wall = mesh_params["nx_wall"]
    nr_wall = mesh_params["nr_wall"]
    nx_wick = mesh_params["nx_wick"]
    nr_wick = mesh_params["nr_wick"]
    nx_vc = mesh_params["nx_vc"]
    nr_vc = mesh_params["nr_vc"]
    
    # Verify that x-direction cell counts match (as required by the mesh)
    if nx_wall != nx_wick or nx_wick != nx_vc:
        print("Warning: Adjusting x-direction cell counts to match nx_wall")
        nx_wick = nx_wall
        nx_vc = nx_wall
    
    # Calculate the total number of cells in x and y directions
    nx = nx_wall  # All sections have same x-resolution
    ny = nr_wall + nr_wick + nr_vc
    
    # Cell size in x-direction (uniform)
    dx = L_total / nx
    
    # Calculate the width of each region in y-direction
    wall_thickness = dimensions["R_wall"] - dimensions["R_wick"]
    wick_thickness = dimensions["R_wick"] - dimensions["R_vc"]
    core_radius = dimensions["R_vc"]
    
    # Create a uniform mesh first
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=1.0)  # We'll adjust y coordinates later
    
    # Get cell centers and adjust the y-coordinates
    x, y = mesh.cellCenters
    
    # Calculate cell heights for each region
    dy_vc = core_radius / nr_vc
    dy_wick = wick_thickness / nr_wick
    dy_wall = wall_thickness / nr_wall
    
    # Create a cell type indicator (0: vapor core, 1: wick, 2: wall)
    cell_types = CellVariable(name="Cell Types", mesh=mesh, value=-1)
    
    # Adjust y-coordinates based on cell position
    new_y = np.zeros_like(y)
    
    for i in range(nx):
        for j in range(ny):
            cell_index = i * ny + j
            
            if j < nr_vc:
                # Vapor core region
                new_y[cell_index] = j * dy_vc + dy_vc/2
                cell_types.value[cell_index] = 0
            elif j < nr_vc + nr_wick:
                # Wick region
                j_local = j - nr_vc
                new_y[cell_index] = core_radius + j_local * dy_wick + dy_wick/2
                cell_types.value[cell_index] = 1
            else:
                # Wall region
                j_local = j - (nr_vc + nr_wick)
                new_y[cell_index] = core_radius + wick_thickness + j_local * dy_wall + dy_wall/2
                cell_types.value[cell_index] = 2
    
    # Create a new mesh with the adjusted coordinates
    from fipy.meshes.nonUniformGrid2D import NonUniformGrid2D
    
    # Need to calculate vertices from cell centers
    nx, ny = mesh.shape
    
    # Create vertices arrays (one more in each dimension than cells)
    vertices_x = np.zeros(nx + 1)
    vertices_y = np.zeros(ny + 1)
    
    # Set x vertices based on uniform spacing
    dx = L_total / nx
    for i in range(nx + 1):
        vertices_x[i] = i * dx
    
    # Set y vertices based on the three different regions
    vertices_y[0] = 0.0  # Bottom boundary
    
    # Vapor core vertices
    for j in range(1, nr_vc + 1):
        vertices_y[j] = j * dy_vc
        
    # Wick vertices
    for j in range(nr_vc + 1, nr_vc + nr_wick + 1):
        j_local = j - nr_vc
        vertices_y[j] = core_radius + j_local * dy_wick
        
    # Wall vertices
    for j in range(nr_vc + nr_wick + 1, ny + 1):
        j_local = j - (nr_vc + nr_wick)
        vertices_y[j] = core_radius + wick_thickness + j_local * dy_wall
    
    # Create the non-uniform grid using vertex coordinates
    new_mesh = NonUniformGrid2D(
        dx=vertices_x[1:] - vertices_x[:-1], 
        dy=vertices_y[1:] - vertices_y[:-1],
        nx=nx, ny=ny
    )
    
    # Recreate the cell types on the new mesh
    new_cell_types = CellVariable(name="Cell Types", mesh=new_mesh, value=-1)
    
    # Fill in the cell types on the new mesh
    for i in range(nx):
        for j in range(ny):
            cell_index = i * ny + j
            
            if j < nr_vc:
                new_cell_types.value[cell_index] = 0  # Vapor core
            elif j < nr_vc + nr_wick:
                new_cell_types.value[cell_index] = 1  # Wick
            else:
                new_cell_types.value[cell_index] = 2  # Wall
    
    return new_mesh, new_cell_types


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

    # Visualize the mesh
    import matplotlib.pyplot as plt

    # Extract the coordinates of the cell centers
    x, y = mesh.cellCenters

    # Plot the mesh
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, marker='o', color='b', s=10)
    plt.title('2D Cartesian Mesh')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
