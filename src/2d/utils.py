from fipy import Grid2D


def preview_mesh(mesh, title="2D Mesh Preview"):
    """
    Visualizes the mesh using matplotlib.

    Parameters:
        mesh: The FiPy mesh object to visualize.
        title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt

    print("Mesh generated:")
    print("Number of cells:", mesh.numberOfCells)
    print("Cell size dx:", mesh.dx)
    print("Cell size dy:", mesh.dy)
    
    # Extract the coordinates of the cell centers
    x, y = mesh.cellCenters

    # Plot the mesh
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, marker='o', color='b', s=10)
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()



def preview_face_mask(mesh, mask, title="Mesh with Face Masks"):
    """
    Overlays the face-based mask on top of the mesh plot.

    Parameters:
        mesh: The FiPy mesh object.
        mask: The FaceVariable containing mask values defined on faces.
        title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt
    
    # First, plot the mesh using cell centers for context
    x_cells, y_cells = mesh.cellCenters
    plt.figure(figsize=(10, 5))
    plt.scatter(x_cells, y_cells, marker='o', color='lightgray', s=10, label='Cells')

    # Extract the face centers to locate the mask values.
    x_faces, y_faces = mesh.faceCenters

    # Get the mask values corresponding to each face.
    mask_values = mask.value

    # We can plot only the faces where the mask is non-zero
    nonzero = mask_values != 0

    # Use a colormap to show different mask values (e.g., 1 vs 2).
    # You could use a fixed color mapping if only a few values exist.
    plt.scatter(x_faces[nonzero], y_faces[nonzero],
                marker='s', s=50, c=mask_values[nonzero],
                cmap='viridis', edgecolor='k', label='Mask')

    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.colorbar(label='Mask Value')
    plt.legend()
    plt.show()