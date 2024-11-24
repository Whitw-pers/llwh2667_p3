import numpy as np
from scipy.sparse import lil_matrix

# Function to create a visibility matrix from inlier data
def GetVmatrix(All_Inlier):
    """
    GetVmatrix: Extracts and combines u, v coordinates, Point IDs, and Camera IDs from inlier data.
    
    Parameters:
    - All_Inlier: DataFrame of inliers, where each row contains PointID, (u, v) coordinates, and CameraID.
    
    Returns:
    - Vmatrix: Combined matrix of u, v coordinates, Point IDs, and Camera IDs.
    """
    All_Inlier = np.array(All_Inlier)
    # Extract each component from the array
    PointIDs = All_Inlier[:, 0]  # First column: Point IDs
    u_coords = All_Inlier[:, 1]  # Second column: u (horizontal pixel coordinates)
    v_coords = All_Inlier[:, 2]  # Third column: v (vertical pixel coordinates)
    CameraIDs = All_Inlier[:, 3]  # Fourth column: Camera IDs

    # Concatenate u, v, PointID, and CameraID into a single matrix
    Vmatrix = np.column_stack((PointIDs, CameraIDs, u_coords, v_coords))

    return Vmatrix

# Function to build the visibility matrix for bundle adjustment
def BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices):
    """
    BuildVisibilityMatrix: Constructs a sparse visibility matrix for bundle adjustment.
    This matrix indicates which parameters affect each observation in the optimization process.
    
    Parameters:
    - n_cameras: Total number of cameras.
    - n_points: Total number of 3D points.
    - camera_indices: Array of indices indicating which camera observes each 2D point.
    - point_indices: Array of indices indicating which 3D point corresponds to each 2D point.
    
    Returns:
    - A: Sparse visibility matrix in lil_matrix format.
    """
    
    # Calculate the number of observations (2D points), each observation contributes two rows (u and v)
    n_observations = len(camera_indices)

    # Calculate the number of parameters (unknowns) in the optimization
    # Each camera has 7 parameters (3 for translation, 4 for rotation as quaternion)
    # Each 3D point has 3 parameters (X, Y, Z)
    n_camera_params = n_cameras * 7
    n_point_params = n_points * 3
    n_params = n_camera_params + n_point_params  # Total number of parameters


    # Initialize a sparse matrix in 'list of lists' (lil) format for efficient row operations
    A = lil_matrix((n_observations * 2, n_params), dtype=int)  # 2 rows per observation (u and v)
    # Create an array of observation indices
    obs_indices = np.arange(n_observations)
    # Fill in the visibility matrix for camera parameters
    for obs_idx, cam_idx in enumerate(camera_indices):
        A[2 * obs_idx, cam_idx * 7:(cam_idx + 1) * 7] = 1  # u coordinate affects camera parameters
        A[2 * obs_idx + 1, cam_idx * 7:(cam_idx + 1) * 7] = 1  # v coordinate affects camera parameters


    # Fill in the visibility matrix for 3D point parameters
    for obs_idx, point_idx in enumerate(point_indices):
        A[2 * obs_idx,
        n_camera_params + point_idx * 3:n_camera_params + (point_idx + 1) * 3] = 1  # u affects point params
        A[2 * obs_idx + 1,
        n_camera_params + point_idx * 3:n_camera_params + (point_idx + 1) * 3] = 1  # v affects point params

    return A  # Return the sparse visibility matrix
