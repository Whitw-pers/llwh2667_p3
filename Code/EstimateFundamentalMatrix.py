from numpy import linalg as LA
import numpy as np
import cv2

'''
def getT(points):
    # find centroid and shift points
    centroid = np.mean(points, axis=0)
    centeredPoints = points - centroid

    # compute scale factor
    dist = np.sqrt(np.sum(centeredPoints ** 2, axis=1))
    scale = np.sqrt(2) / np.mean(dist)

    # construct normalization matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    return T
'''

def getT(points):
    '''
    https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/arao83/index.html
    '''
    n = points.shape[0]
    C = np.mean(points, axis=0)

    # center points
    u = points[:, 0] - C[0]
    v = points[:, 1] - C[1]

    # find stddev for centered points
    std_u = 1 / np.std(u)
    std_v = 1 / np.std(v)

    # find transformation matrix
    T_scale = np.array([[std_u, 0, 0],
                        [0, std_v, 0],
                        [0, 0, 1]])
    T_offset = np.array([[1, 0, -C[0]],
                         [0, 1, -C[1]],
                         [0, 0, 1]])
    T = T_scale @ T_offset

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

    '''
    # Convert input data from DataFrames to numpy arrays for easier manipulation
    x1DF = x1DF.to_numpy()
    x2DF = x2DF.to_numpy()
    x1_aug = np.hstack((x1DF, np.ones((x1DF.shape[0], 1))))
    x2_aug = np.hstack((x2DF, np.ones((x2DF.shape[0], 1))))

    # Step 1: Normalization
    T1 = getT(x1DF)
    T2 = getT(x2DF)
    x1_norm = (T1 @ x1_aug.T).T
    x2_norm = (T2 @ x2_aug.T).T

    # Step 2: Construct the Linear System (A)
    A = np.zeros((x1DF.shape[0], 9))
    for i in range(x1DF.shape[0]):
        xS = x1_norm[i, 0]
        yS = x1_norm[i, 1]
        xT = x2_norm[i, 0]
        yT = x2_norm[i, 1]
        A[i] = [xS * xT, xS * yT, xS, yS * xT, yS * yT, yS, xT, yT, 1]

    # Step 3: Solve the Linear System Using SVD
    U, S, Vh = LA.svd(A)
    f = Vh[-1, :]
    F = f.reshape((3, 3)).T
    
    # Step 4: Enforce Rank-2 Constraint on F
    U, S, Vh = LA.svd(F)
    S[-1] = 0  # Set the smallest singular value to zero
    F = U @ (np.diag(S) @ Vh)

    # Step 6: Denormalize the Fundamental Matrix
    F = T2.T @ (F @ T1)

    # Step 5: Scaling - Ensure the last element of F is 1 for consistency
    F = F / F[2, 2]  # Corrected to F[2, 2] for the bottom-right element

    '''
    x1DF = x1DF.to_numpy()
    x2DF = x2DF.to_numpy()
    F, mask = cv2.findFundamentalMat(x1DF, x2DF, cv2.FM_LMEDS)
    

    return F  # Return the estimated Fundamental matrix




