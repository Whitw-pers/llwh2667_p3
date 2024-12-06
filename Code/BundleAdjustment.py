import numpy as np
import sys
import pandas as pd

from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix  # Ensure this import if you use lil_matrix for the visibility matrix

# Function to perform Bundle Adjustment to optimize camera poses and 3D point positions
def BundleAdjustment(Call, Rall, Xall, K, sparseVmatrix, n_cameras, n_points, camera_indices, point_indices, xall):
    """
    BundleAdjustment: Refines camera poses and 3D point positions to minimize the reprojection error
    for a set of cameras and 3D points using non-linear optimization.
    
    Parameters:
    - Call: List of initial camera positions (list of 3x1 arrays).
    - Rall: List of initial rotation matrices for each camera (list of 3x3 arrays).
    - Xall: DataFrame of 3D points with IDs (Nx4).
    - K: Intrinsic camera matrix (3x3).
    - sparseVmatrix: Sparse matrix for Jacobian sparsity pattern to speed up optimization.
    - n_cameras: Number of cameras.
    - n_points: Number of 3D points.
    - camera_indices: Indices indicating which camera observes each 2D point.
    - point_indices: Indices indicating which 3D point corresponds to each 2D point.
    - xall: Array of observed 2D points in image coordinates.
    
    Returns:
    - CoptAll: List of optimized camera positions (3x1 arrays).
    - RoptAll: List of optimized rotation matrices (3x3 arrays).
    - XoptAll: DataFrame of optimized 3D points with IDs.
    """
    
    def reprojection_loss(x, n_cameras, n_points, camera_indices, point_indices, xall, K):
        """
        Computes the reprojection error for the current estimates of camera poses and 3D points.
        
        Parameters:
        - x: Flattened array containing all camera positions, orientations (as quaternions), and 3D points.
        - n_cameras: Number of cameras.
        - n_points: Number of 3D points.
        - camera_indices: Indices indicating which camera observes each 2D point.
        - point_indices: Indices indicating which 3D point corresponds to each 2D point.
        - xall: Observed 2D points (Nx2).
        - K: Intrinsic camera matrix (3x3).
        
        Returns:
        - residuals: Flattened array of reprojection errors for all points across all cameras.
        """
        
        I = np.eye(3)  # Identity matrix for projection matrix construction
        
        # First camera pose (fixed as the reference)
        C0 = np.zeros((3, 1))  # Assume the first camera position is at the origin
        R0 = np.eye(3)  # Assume the first camera has no rotation
        P0 = np.matmul(np.matmul(K, R0), np.concatenate((I, -C0), axis=1))  # Projection matrix for the first camera
        
        Ps = [P0]  # Initialize list of projection matrices with the first camera's projection matrix
        
        # Reconstruct the remaining camera poses from the optimization variable `x`
        # Extract camera parameters (excluding the first camera)
        camera_params = x[:7 * (n_cameras - 1)].reshape((-1, 7))

        for params in camera_params:
            C = params[:3][:, None]  # Translation vector as column
            q = params[3:]  # Quaternion
            R = Rotation.from_quat(q).as_matrix()
            P = K @ np.concatenate((R, -R @ C), axis=1)
            Ps.append(P)

        # Collect the projection matrices based on the camera indices for each observation
        Pall = np.array([Ps[int(idx)] for idx in camera_indices])
        
        # Extract and reshape the 3D points from the optimization variable `x`
        X = x[7 * (n_cameras - 1):].reshape((-1, 3))
        
        # Collect 3D points in homogeneous coordinates based on point indices for each observation
        Xall = np.array([np.pad(X[int(idx)], (0, 1), constant_values=1)[:, None] for idx in point_indices])

        # Projection of 3D points onto the image plane
        x_proj = np.squeeze(np.matmul(Pall, Xall))  # Projected 2D points in homogeneous coordinates [x, y, z]
        x_proj = x_proj / x_proj[:, 2, None]  # Normalize to get pixel coordinates [u, v, 1]
        x_proj = x_proj[:, :2]  # Extract [u, v] coordinates
        
        # Calculate the reprojection error as the difference between observed and projected points
        reprojection_error = x_proj - xall

        residuals = reprojection_error.ravel()
        return residuals
    
    print("\n Running BA.....")

    # Initial parameters setup

    # Concatenate initial camera positions and orientations (as quaternions) into a single parameter vector `init_x`
    init_camera_params = []
    for idx, (Ci, Ri) in enumerate(zip(Call, Rall)):
                
        # Convert rotation matrix to quaternion for initialization
        # Quaternion representation of the rotation matrix (4,)
        # scipy's from_matrix returns Rotation; as_quat() returns (x, y, z, w)
        q = Rotation.from_matrix(Ri).as_quat()

        # For the first camera, do not add to init_x (fixed)
        if idx == 0:
            # Just store for reference, no addition to init_x
            pass
        else:
            # Append [C_x, C_y, C_z, q_x, q_y, q_z, q_w]
            cam_params = np.hstack((Ci.flatten(), q))
            init_camera_params.append(cam_params)

    init_camera_params = np.array(init_camera_params).ravel() if len(init_camera_params) > 0 else np.array([])

    # Flatten the initial 3D points and add them to the parameter vector `init_x`
    # Xall assumed to have columns: [PointID, X, Y, Z]
    point_ids = Xall.iloc[:, 0].values.astype(int)

    # Extract [X, Y, Z] coordinates from the 3D points
    X_points = Xall.iloc[:, 1:].values
    init_point_params = X_points.ravel()  # Flatten 3D points

    # Extract point IDs for future use
    # Append flattened 3D points to `init_x`
    # Combine camera and point parameters into one vector
    init_x = np.hstack((init_camera_params, init_point_params))

    # Perform bundle adjustment using non-linear least squares optimization
    res = least_squares(
        reprojection_loss,
        init_x,
        jac_sparsity=sparseVmatrix,
        args=(n_cameras, n_points, camera_indices, point_indices, xall, K),
        method='trf',
        verbose=2
    )
    # Extract optimized camera poses from the solution
    optimized_x = result.x
    optimized_camera_params = optimized_x[:7 * (n_cameras - 1)].reshape((n_cameras - 1, 7))
    optimized_points = optimized_x[7 * (n_cameras - 1):].reshape((n_points, 3))

    # Reconstruct camera poses
    CoptAll = []
    RoptAll = []
    # First camera fixed
    CoptAll.append(np.zeros((3, 1)))
    RoptAll.append(np.eye(3))

    for i in range(1, n_cameras):
        cam_p = optimized_camera_params[i - 1]
        C = cam_p[:3][:, None]
        q = cam_p[3:]  # [qx, qy, qz, qw]
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(q).as_matrix()

        CoptAll.append(C)
        RoptAll.append(R)


    # Extract optimized 3D points and combine with IDs

        # Combine optimized points with their IDs
        # Original Xall: [PointID, X, Y, Z]
        # We now have optimized X, Y, Z
        # We'll reassemble into the same format [PointID, X, Y, Z]
        XoptAll = np.hstack((Xall[:, :1], optimized_points))

    return CoptAll, RoptAll, XoptAll  # Return optimized camera positions, rotation matrices, and 3D points
