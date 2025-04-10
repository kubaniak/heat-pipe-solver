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

from fipy import CellVariable
from fipy.meshes.nonUniformGrid2D import NonUniformGrid2D
import numpy as np

def generate_composite_mesh(mesh_params, dimensions):
    """
    Generate a composite FiPy mesh for a cylindrical heat pipe
    with variable radial resolution (vapor core, wick, wall).
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

    # Assign cell types
    ny = sum(nr.values())
    cell_types = np.zeros(mesh.numberOfCells, dtype=int)
    split1 = nr["vc"]
    split2 = split1 + nr["wick"]

    for i in range(nx):
        base = i * ny
        cell_types[base + split1:base + split2] = 1  # wick
        cell_types[base + split2:base + ny] = 2      # wall

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
