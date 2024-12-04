import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# Function to perform Non-linear Perspective-n-Point (PnP) optimization to refine the camera pose
def NonlinearPnP(Xs, xs, K, Cnew, Rnew):
    """
    NonlinearPnP: Refines the camera pose (position and orientation) using non-linear optimization
    to minimize the reprojection error between observed 2D points and projected 3D points.
    
    Parameters:
    - Xs: DataFrame of 3D points in world coordinates (Nx4 with IDs).
    - xs: DataFrame of corresponding 2D points in image coordinates (Nx3 with IDs).
    - K: Intrinsic camera matrix (3x3).
    - Cnew: Initial guess for camera position (3x1).
    - Rnew: Initial guess for camera rotation matrix (3x3).
    
    Returns:
    - Copt: Optimized camera position (3x1).
    - Ropt: Optimized rotation matrix (3x3).
    """
    
    def reprojection_loss(x, Xset, xset, K):
        """
        Computes the reprojection error for the current camera pose estimate.
        
        Parameters:
        - x: Flattened array containing the camera position and rotation as a quaternion.
        - Xset: 3D points in world coordinates (Nx3).
        - xset: Corresponding 2D image points (Nx2).
        - K: Intrinsic camera matrix (3x3).
        
        Returns:
        - residuals: Flattened array of reprojection errors for each point.
        """
        
        # Extract the camera translation vector (position) from the optimization variable x
        C = x[:3][:, None]  # Camera position (3x1)
        
        # Convert the quaternion (x[3:]) to a rotation matrix R (3x3)
        R = Rotation.from_quat(x[3:]).as_matrix()
        
        # Construct the projection matrix P using the rotation and translation
        I = np.eye(3)
        P = np.matmul(np.matmul(K, R), np.concatenate((I, -C), axis=1))
        
        # Prepare the 3D points in homogeneous coordinates
        Xset = np.pad(Xset, ((0, 0), (0, 1)), constant_values=1).T  # [X, Y, Z, 1] format
        
        # Project the 3D points Xset into the 2D image plane using the projection matrix P
        x_proj = np.matmul(P, Xset).T  # Projected 2D points in homogeneous coordinates [x, y, z]
        x_proj = x_proj / x_proj[:, 2, None]  # Normalize to get pixel coordinates [u, v, 1]
        x_proj = x_proj[:, :2]  # Extract [u, v] coordinates
        
        # Calculate the reprojection error as the difference between observed and projected points
        residuals = (xset - x_proj).ravel()  # Flatten the error array for least_squares
        
        return residuals
        
    # Convert initial rotation matrix Rnew to a quaternion representation
    R = Rotation.from_matrix(Rnew)
    Q = R.as_quat()
    
    # Initial parameters for optimization: flatten camera position and convert rotation to quaternion
    # Initial camera position as a 1D array (3,)
    # Initial rotation as a quaternion (4,)
    # Combine position and quaternion for optimization
    x = np.append(Cnew, Q)
    
    # Prepare the 3D points and 2D points for optimization
    # Extract [X, Y, Z] coordinates from the 3D points
    # Extract [u, v] coordinates from the 2D points
    Xset = Xs.iloc[:, 1:]
    xset = xs.iloc[:, 1:]
    Xset = Xset.to_numpy()
    xset = xset.to_numpy()
    
    # Run non-linear optimization to minimize reprojection error
    solution = least_squares(fun = reprojection_loss, x0 = x, args = (Xset, xset, K))
    solution = solution.x
    
    # Extract the optimized camera position and rotation matrix from the solution
    # Optimized camera position (3x1)
    # Optimized rotation matrix (3x3)
    Ropt = Rotation.from_quat(solution[3:]).as_matrix()
    Copt = solution[:3]

    return Copt, Ropt  # Return the optimized camera position and rotation