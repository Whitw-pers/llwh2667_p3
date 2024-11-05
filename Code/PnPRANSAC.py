import numpy as np
import random
from tqdm import tqdm
from numpy import linalg as LA

# Function to calculate the reprojection error of a 3D point projected onto the image plane
def CalReprojErr(X, x, P):
    """
    CalReprojErr: Computes the reprojection error for a 3D point X when projected
    onto the image plane with a given camera matrix P.
    
    Parameters:
    - X: 3D point in homogeneous coordinates with an ID (4x1).
    - x: Observed 2D point in the image with an ID (3x1).
    - P: Projection matrix (3x4).
    
    Returns:
    - e: Reprojection error (scalar).
    """
    u = x[1]  # x-coordinate in the image
    v = x[2]  # y-coordinate in the image

    # Convert 3D point X to homogeneous coordinates without ID
    X_noID = np.concatenate((X[1:], np.array([1])), axis=0)
    
    # Extract rows of P for computation
    P1 = P[0, :]
    P2 = P[1, :]
    P3 = P[2, :]
    
    # Calculate reprojection error
    e = (u - np.matmul(P1, X_noID) / np.matmul(P3, X_noID))**2 + \
        (v - np.matmul(P2, X_noID) / np.matmul(P3, X_noID))**2
    
    return e

# Function to estimate the camera projection matrix using Linear Perspective-n-Point (PnP)
def LinearPnP(X, x, K):
    """
    LinearPnP: Computes the camera projection matrix (P) given a set of 3D points (X)
    and their corresponding 2D projections (x) using a linear approach.
    
    Parameters:
    - X: DataFrame of 3D points with IDs (Nx4).
    - x: DataFrame of corresponding 2D points with IDs (Nx3).
    - K: Intrinsic camera matrix (3x3).
    
    Returns:
    - P: Camera projection matrix (3x4).
    """
    X = X.to_numpy()
    x = x.to_numpy()

    # Construct the linear system A from the correspondences
    
    
    # Solve the linear system using Singular Value Decomposition (SVD)
    # Last column of V gives the solution for P
    
    return P

# Function to perform PnP using RANSAC to find the best camera pose with inliers
def PnPRANSAC(Xset, xset, K, M=2000, T=10):
    """
    PnPRANSAC: Performs Perspective-n-Point (PnP) with RANSAC to robustly estimate the
    camera pose (position and orientation) from 2D-3D correspondences.
    
    Parameters:
    - Xset: DataFrame of 3D points with IDs (Nx4).
    - xset: DataFrame of corresponding 2D points with IDs (Nx3).
    - K: Intrinsic camera matrix (3x3).
    - M: Number of RANSAC iterations (default: 2000).
    - T: Threshold for reprojection error to count as an inlier (default: 10).
    
    Returns:
    - Cnew: Estimated camera center (3x1).
    - Rnew: Estimated rotation matrix (3x3).
    - Inlier: List of inlier 3D points.
    """
    
    
    # List to store the largest set of inliers
    # Total number of correspondences
    
    for i in tqdm(range(M)):
        # Randomly select 6 2D-3D pairs
                
        # Extract subsets of 3D and 2D points

        # Estimate projection matrix P using LinearPnP with the selected points
        
        # Calculate inliers by checking reprojection error for all points
            
            # Calculate reprojection error
            
            # If error is below threshold T, consider it as an inlier
        
        # Update inlier set if the current inlier count is the highest found so far
        
    # Decompose Pnew to obtain rotation R and camera center C
    
    # Enforce orthogonality of R
    
    return Cnew, Rnew, Inlier  # Return the estimated camera center, rotation matrix, and inliers
