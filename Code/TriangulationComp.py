import numpy as np
import matplotlib.pyplot as plt
import os

def TriangulationComp(Cset, Rset, Xset, SAVE_DIR, filename):
    """
    PlotCameraPts: Visualizes two camera poses with orientations and triangulated 3D points.

    Arguments:
    - Cset: List of camera centers for each pose (2x3x1 list).
    - Rset: List of rotation matrices for each pose (2x3x3 list).
    - Xset: 3D list where:
            - First dimension: Half (1 to 2)
            - Second dimension: Rows contain individual points
            - Third dimension: Columns [ID, x, y, z]
    - SAVE_DIR: Directory where the plot will be saved.
    - filename: Name of the file for the saved plot (e.g., 'TwoCameraPose.png').
    """
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each half of 3D points with a different color
    colors = ['blue', 'orange']  # Adjust for two sets instead of four
    for half, color in enumerate(colors, start=1):
        # Extract the points for the current half
        half_points = np.array(Xset[half - 1])  # Convert to numpy array for easier slicing
        X = half_points[:, 1]  # X-coordinate (column index 1)
        Z = half_points[:, 3]  # Z-coordinate (column index 3)
        plt.scatter(X, Z, color=color, label=f'Set {half}', alpha=0.4, s=2)

    # Define color mapping for the camera poses
    pose_colors = ['blue', 'orange']

    # Extract camera positions and plot orientations
    scale = 2  # Scale for the orientation vectors

    for i, (C, R) in enumerate(zip(Cset, Rset)):
        # C is 3x1; take its x and z components
        cx, cz = C[0][0], C[2][0]

        # Plot the orientation vectors for the camera's local axes
        z_axis = R[2, :]  # Third row of R (local z-axis)

        # Start at the camera position
        start = np.array([cx, cz])

        # Compute end points of the Z-axis vector
        end_z = start + scale * np.array([z_axis[0], z_axis[2]])

        # Plot the Z-axis orientation vector with different colors for each camera pose
        plt.arrow(start[0], start[1], end_z[0] - start[0], end_z[1] - start[1],
                  color=pose_colors[i], head_width=0.2, label=f'Camera {i + 1} Local Z Axis' if i == 0 else "", alpha=0.6)

    # Add plot labels and legend
    plt.xlabel('X position')
    plt.ylabel('Z position')
    plt.title('Camera Poses with Orientations and 3D Points (X vs Z)')
    plt.legend()
    plt.grid(True)

    # Set the limits to be fixed for both axes
    plt.xlim(-15, 15)
    plt.ylim(-20, 20)

    # Show the plot on screen
    # plt.show()

    # Save the plot to the specified directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)  # Create the directory if it doesn't exist
    output_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


# Example usage:
# Xset = [
#     [[1, 0.1, 0.2, 0.3], [2, 0.4, 0.5, 0.6], ...],  # Points for Half 1
#     [[11, 1.1, 1.2, 1.3], [12, 1.4, 1.5, 1.6], ...],  # Points for Half 2
# ]
# Cset = [
#     np.array([[0], [0], [0]]),  # Camera 1 position
#     np.array([[1], [1], [1]]),  # Camera 2 position
# ]
# Rset = [
#     np.eye(3),  # Camera 1 rotation matrix
#     np.eye(3),  # Camera 2 rotation matrix
# ]
# PlotCameraPts(Cset, Rset, Xset, 'output_directory', 'TwoCameraPose.png')