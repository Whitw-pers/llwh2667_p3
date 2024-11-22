import numpy as np
import matplotlib.pyplot as plt
import os

def PlotPtsCams(Cset, Rset, Xset, SAVE_DIR, filename):
    """
    PlotCameraPts: Visualizes a single camera pose with orientation and triangulated 3D points.

    Arguments:
    - Cset: List containing one camera center (1x3x1 list).
    - Rset: List containing one rotation matrix (1x3x3 list).
    - Xset: 3D list where:
            - Rows contain individual points
            - Columns [ID, x, y, z].
    - SAVE_DIR: Directory where the plot will be saved.
    - filename: Name of the file for the saved plot (e.g., 'CameraPose.png').
    """
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Extract the points
    Xset = np.array(Xset[0])  # Use the single element in the list
    X = Xset[:, 1]  # X-coordinate (column index 1)
    Z = Xset[:, 3]  # Z-coordinate (column index 3)
    plt.scatter(X, Z, color='blue', label='3D Points', alpha=0.7, s=3)

    # Extract the single camera position and rotation matrix
    C = Cset[0]  # Single camera position (3x1)
    R = Rset[0]  # Single rotation matrix (3x3)

    # Define scale for the orientation vector
    scale = 3

    # Plot the camera orientation vector
    cx, cz = C[0][0], C[2][0]  # Extract X and Z components of the camera position
    z_axis = R[2, :]  # Third row of R (local z-axis)

    # Start at the camera position
    start = np.array([cx, cz])

    # Compute end points of the Z-axis vector
    end_z = start + scale * np.array([z_axis[0], z_axis[2]])

    # Plot the Z-axis orientation vector
    plt.arrow(start[0], start[1], end_z[0] - start[0], end_z[1] - start[1],
              color='blue', head_width=0.9, label='Camera Local Z Axis', alpha=0.6)

    # Add plot labels and legend
    plt.xlabel('X position')
    plt.ylabel('Z position')
    plt.title('Camera Pose with Orientation and 3D Points (X vs Z)')
    plt.legend()
    plt.grid(True)

    # Set the limits to be fixed for both axes
    plt.xlim(-15, 15)
    plt.ylim(-20, 20)

    # Save the plot to the specified directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)  # Create the directory if it doesn't exist
    output_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")
