"""
mesh.py

This module contains functions to generate 2D meshes using FiPy.
For now, we generate a simple Cartesian grid using Grid2D.
"""

from fipy import Grid2D

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

# For testing this module independently (optional)
if __name__ == '__main__':
    # Example parameters: 0.5 m axial, 0.009 m lateral, 500 cells in x, 9 cells in y.
    # params = get_all_params()  # Ensure that the parameters are loaded
    mesh = generate_mesh_2d(0.5, 0.009, 500, 9)
    print("Mesh generated:")
    print("Number of cells:", mesh.numberOfCells)
    print("Cell size dx:", mesh.dx)
    print("Cell size dy:", mesh.dy)
    
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
