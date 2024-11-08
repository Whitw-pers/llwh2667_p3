import numpy as np
from numpy import linalg as LA
import math
import sys

def getT(points):
    # find centriod and shift points
    centroid = np.mean(points, axis=0)
    centeredPoints = points - centroid

    # compute scale factor
    dist = np.sqrt(np.sum(centeredPoints ** 2, 1))
    scale = np.sqrt(2) / np.mean(dist)

    # construct normalization matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    return T

def EstimateFundamentalMatrix(x1DF, x2DF):
    """
    Estimates the Fundamental matrix (F) between two sets of points using the 8-point algorithm,
    with normalization for numerical stability.

    Args:
        x1DF (DataFrame): Source image points (x, y).
        x2DF (DataFrame): Target image points (x, y).

    Returns:
        F (ndarray): The estimated 3x3 Fundamental matrix.
    """
    # Convert input data from DataFrames to numpy arrays for easier manipulation
    x1DF = x1DF.to_numpy()
    x2DF = x2DF.to_numpy()
    x1_aug = np.hstack((x1DF, np.ones((x1DF.shape[0], 1))))
    x2_aug = np.hstack((x2DF, np.ones((x2DF.shape[0], 1))))
    
    # Step 1: Normalization - Improves numerical stability by normalizing points to a common scale
        # Compute centroids of the points in each image
        # Shift points to have zero mean (centering) by subtracting the centroid
        # Compute scaling factors to make the mean distance of points from the origin equal to sqrt(2)
        # Construct the normalization transformation matrices for both images
    T1 = getT(x1_aug)
    T2 = getT(x2_aug)
    x1_norm = T1 @ x1_aug.T
    x2_norm = T2 @ x2_aug.T

    # Step 2: Construct the Linear System (A) for Estimating F
        # Set up a linear system A*f = 0, where f is the vector form of the matrix F
        # Initialize the first row of matrix A using the first pair of points
        # Normalize the source point using the transformation matrix
    x1 = x1_norm / x1_norm[2]  # Convert to homogeneous coordinates
        # Normalize the target point using the transformation matrix
    x2 = x2_norm / x2_norm[2]  # Convert to homogeneous coordinates
        # Construct the first row of A using the epipolar constraint x2^T * F * x1 = 0
        # Repeat for all other points to build the full matrix A
    A = np.zeros((x1DF.shape[0], 9))
    for i in range(x1DF.shape[0]):
        xS, yS = x1[:, i][:2]
        xT, yT = x2[:, i][:2]
        A[i] = [xS * xT, xS * yT, xS, yS * xT, yS * yT, yS, xT, yT, 1]

    # Step 3: Solve the Linear System Using SVD
    # Find the solution to A*f = 0 by taking the SVD of A and using the right singular vector corresponding to the smallest singular value
    # Hint: u, s, vh = LA.svd(A)
    # F is the last column of V (smallest singular value)
    # Reshape F into a 3x3 matrix
    U, S, Vh = LA.svd(A)
    f = Vh[-1, :]
    F = f.reshape((3, 3)).T

    # Step 4: Enforce Rank-2 Constraint on F
    # The Fundamental matrix must be rank-2, so we set the smallest singular value of F to zero
    # Set the smallest singular value to zero
    # Reconstruct F with rank-2 constraint
    U, S, Vh = LA.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vh

    # Step 5: Scaling - Ensure the last element of F is 1 for consistency
    F = F / F[3, 3]

    # Step 6: Denormalize the Fundamental Matrix
    # Transform F back to the original coordinate system using the inverse of the normalization transformations
    F = T2.T @ F @ T1

    return F  # Return the estimated Fundamental matrix
