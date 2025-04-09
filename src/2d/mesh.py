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
from fipy import Grid2D

def generate_composite_mesh(mesh_params, dimensions):
    r"""
    Generates a composite 2D mesh for a heat pipe using FiPy.
    
    The mesh is constructed in the (x,r) (or equivalently (x,y)) plane.
    In this configuration, the x‑axis is the axial (length) direction, while
    the y‑axis represents the radial direction (from the vapor core at y = 0 up
    to the outer wall at y = R_wall). The three layers (vapor core, wick, and wall)
    are prescribed by different numbers of cells in the y‑direction.
    
    Parameters
    ----------
    mesh_params : dict
        Dictionary with the following keys:
            "nx_wall": number of cells in x for the wall layer (must equal others),
            "nr_wall": number of cells in r (y) for the wall layer,
            "nx_wick": number of cells in x for the wick layer,
            "nr_wick": number of cells in r (y) for the wick layer,
            "nx_vc": number of cells in x for the vapor core,
            "nr_vc": number of cells in r (y) for the vapor core.
    
    dimensions : dict
        Dictionary with the following keys:
            "R_wall": outer radius of the wall,
            "R_wick": outer radius of the wick (inner boundary of wall),
            "R_vc": radius of the vapor core (inner boundary of wick),
            "L_e": axial length of the evaporator,
            "L_a": axial length of the adiabatic (wick) region,
            "L_c": axial length of the condenser.
    
    Returns
    -------
    mesh : fipy.meshes.Grid2D
        A FiPy mesh whose x‑direction spans the entire heat pipe length and whose
        y‑direction is partitioned into the three layers with nonuniform spacing.
    """
    # ------------------------------
    # 1. Axial (x) direction parameters
    # ------------------------------
    L_e = dimensions["L_e"]
    L_a = dimensions["L_a"]
    L_c = dimensions["L_c"]
    L_total = L_e + L_a + L_c
    
    # Check that the axial cell counts are the same for all layers:
    nx_wall = mesh_params["nx_wall"]
    nx_wick = mesh_params["nx_wick"]
    nx_vc   = mesh_params["nx_vc"]
    
    if not (nx_wall == nx_wick == nx_vc):
        raise ValueError("The number of x-cells for each layer must be identical for a conformal mesh.")
    
    nx = nx_wall
    dx = L_total / nx

    # ------------------------------
    # 2. Radial (y) direction parameters
    # ------------------------------
    nr_vc   = mesh_params["nr_vc"]
    nr_wick = mesh_params["nr_wick"]
    nr_wall = mesh_params["nr_wall"]

    mesh_wall = Grid2D(dx=dx, dy=(dimensions["R_wall"] - dimensions["R_wick"]) / nr_wall, nx=nx, ny=nr_wall)
    mesh_wick = Grid2D(dx=dx, dy=(dimensions["R_wick"] - dimensions["R_vc"]) / nr_wick, nx=nx, ny=nr_wick)
    mesh_vc = Grid2D(dx=dx, dy=dimensions["R_vc"] / nr_vc, nx=nx, ny=nr_vc)
    

    mesh_vc_wick = mesh_vc + (mesh_wick + ((0,), (dimensions["R_vc"],)))
    mesh = mesh_vc_wick + (mesh_wall + ((0,), (dimensions["R_wick"],)))

    return mesh


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

    mesh = generate_composite_mesh(mesh_params, dimensions)
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
