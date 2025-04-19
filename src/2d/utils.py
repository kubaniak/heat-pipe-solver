from fipy import Grid2D
from fipy.tools import numerix as npx
from scipy.optimize import curve_fit


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


def preview_cell_types(mesh, cell_types, title="Heat Pipe Regions"):
    """
    Visualizes the different regions of a heat pipe based on cell types.

    Parameters:
        mesh: The FiPy mesh object.
        cell_types: CellVariable containing region markers (0 for vapor core, 1 for wick, 2 for wall).
        title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract cell centers
    x, y = mesh.cellCenters
    
    # Get cell type values
    values = cell_types.value
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Create a scatter plot with different colors for each region
    # Using a custom colormap appropriate for distinct regions
    regions = ['Vapor Core', 'Wick', 'Wall']
    colors = ['lightblue', 'orange', 'gray']
    
    # Plot each region separately for better control and legend
    for i, (region, color) in enumerate(zip(regions, colors)):
        mask = (values == i)
        plt.scatter(x[mask], y[mask], marker='o', color=color, s=10, label=region)
    
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


def init_tripcolor_viewer(mesh):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np

    x = mesh.cellCenters[0]
    y = mesh.cellCenters[1]
    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(6, 4))
    tpc = ax.tripcolor(triang, np.zeros_like(x), shading='gouraud', cmap="inferno")
    cbar = fig.colorbar(tpc, ax=ax, label="Temperature [K]")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()

    return fig, ax, tpc, triang

def compare_tabulated_and_interpolated_data(fit_function, tabulated_data):
    """
    Compare tabulated sodium property data with the fitted symbolic functions by creating a plot.
    """
    import matplotlib.pyplot as plt
    
    # Temperature data points
    T_data = npx.arange(400, 2501, 100)
    # Temperature range for smoother fitted curve
    T_fit = npx.linspace(400, 2500, 1000)
    
    # Create a figure for the plots
    # Calculate fitted values
    fitted_values = npx.array([fit_function(T) for T in T_fit])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(T_data, tabulated_data, 'o', label='Tabulated Data', markersize=5)
    plt.plot(T_fit, fitted_values, '-', label='Fitted Curve', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Property Value')
    plt.title('Comparison of Tabulated and Fitted Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fit_and_generate_symbolic_function(T_data, Y_data, model_func):
    params, _ = curve_fit(model_func, T_data, Y_data)
    def symbolic(T):
        return model_func(T, *params)
    print(f"Fitted parameters: {params}")
    return symbolic


def save_animation(frames_path, output_path, fps=10):
    """
    Compiles frames into a video using OpenCV.

    Parameters:
        frames_path (str): Path to the folder containing frame images.
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the video.
    """
    import os
    import cv2

    # Get the list of frame files and sort them
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".png")])

    # Read the first frame to determine the video size
    first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    height, width, _ = first_frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_path, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")