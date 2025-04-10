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