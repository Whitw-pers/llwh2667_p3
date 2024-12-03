import numpy as np
import random
from tqdm import tqdm
from numpy import linalg as LA
import cv2

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
    X = X.to_numpy()
    x = x.to_numpy()

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

    finally found detailed description of the method:
    https://www.cim.mcgill.ca/~langer/558/2009/lecture18.pdf
    """
    
    X = X.to_numpy()
    x = x.to_numpy()

    A = np.empty((0, 12), np.float32)

    for i in range(len(x)):
        # normalize homogenous 2D points
        xPnt, yPnt = x[i, 1], x[i, 2]
        normPnt = LA.inv(K) @ np.array([[xPnt], [yPnt], [1]])
        normPnt = normPnt / normPnt[2]

        # make corresponding 3D point homogenous
        Xpnt = X[i, 1:4]
        Xpnt = Xpnt.reshape((3, 1))
        Xpnt = np.append(Xpnt, 1)

        # A1 = np.hstack((np.zeros((4,)), -Xpnt.T, normPnt[1] * (Xpnt.T)))
        # A2 = np.hstack((Xpnt.T, np.zeros((4,)), -normPnt[0] * (Xpnt.T)))
        A1 = np.hstack((np.zeros((4,)), Xpnt.T, -normPnt[1] * (Xpnt.T)))
        A2 = np.hstack((Xpnt.T, np.zeros((4,)), -normPnt[0] * (Xpnt.T)))
        #A3 = np.hstack((-normPnt[1] * (Xpnt.T), normPnt[0] * (Xpnt.T), np.zeros((4,))))

        #for a in [A1, A2, A3]:
        #    A = np.append(A, [a], axis = 0)
        for a in [A1, A2]:
            A = np.append(A, [a], axis = 0)

    A = np.float32(A)

    # Solve the linear system using Singular Value Decomposition (SVD)
    U, S, VT = LA.svd(A)

    # Last column of V gives the solution for P
    P = VT.T[:, -1]
    P = P.reshape((3, 4))
    
    # impose orthogonality constraint
    R = P[:, 0:3]
    U, S, VT = LA.svd(R)
    R = U @ VT

    # check determinant sign
    t = P[:, 3]
    if LA.det(R) < 0:
        R = -R
        t = -t

    R = R.reshape((3, 3))
    t = t.reshape((3, 1))
    P = np.hstack((R, t))
    

    '''
    X = X.to_numpy()
    x = x.to_numpy()

    #x = np.hstack((x[:, 1:3], np.ones([len(x), 1])))
    #x = LA.inv(K) @ x.T   # must normalize x by K matrix prior to constructing Ax = 0
    #x = x.T
    for i in range(len(x)):
        x[i, :] = x[i, :] / x[i, 2]

    # Construct the linear system A from the correspondences
    A = np.zeros([2*np.size(X, axis = 0), 12])

    for i in range(np.size(X, axis = 0)):
        ax = np.array([X[i,1], X[i,2], X[i,3], 1, 0, 0, 0, 0, -x[i,0]*X[i,1], -x[i,0]*X[i,2], -x[i,0]*X[i,3], -x[i,0]])
        ay = np.array([0, 0, 0, 0, X[i,1], X[i,2], X[i,3], 1, -x[i,1]*X[i,1], -x[i,1]*X[i,2], -x[i,1]*X[i,3], -x[i,1]])

        A[2*i, :] = ax
        A[(2*i + 1), :] = ay
    
    # Solve the linear system using Singular Value Decomposition (SVD)
    U, S, VT = LA.svd(A)

    # Last column of V gives the solution for P
    P = VT.T[:, -1]
    P = P.reshape((3, 4))
    '''
    
    return P

# Function to perform PnP using RANSAC to find the best camera pose with inliers
def PnPRANSAC(Xset, xset, K, M=2000, T=1000):
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
    
    '''
    # List to store the largest set of inliers
    # Total number of correspondences
    inliersMax = 0
    for i in tqdm(range(M)):
        # Randomly select 6 2D-3D pairs
        randIdxs = np.random.choice(Xset.index, size=6, replace=False)
                
        # Extract subsets of 3D and 2D points
        Xsubset = Xset.loc[randIdxs]
        xsubset = xset.loc[randIdxs]

        # Estimate projection matrix P using LinearPnP with the selected points
        P_est = LinearPnP(Xsubset, xsubset, K)
        
        # Calculate inliers by checking reprojection error for all points 
        inliersIdx = []
        for j in Xset.index:
            # Calculate reprojection error
            error = CalReprojErr(Xset.loc[j], xset.loc[j], P_est)
            
            # If error is below threshold T, consider it as an inlier
            if error < T:
                inliersIdx.append(j)
        
        # Update inlier set if the current inlier count is the highest found so far
        if len(inliersIdx) > inliersMax:
            inliersMax = len(inliersIdx)
            P_new = P_est
            Inlier = Xset.loc[inliersIdx]

    '''
    # utilize openCV to allow continued work on following functions

    X = Xset.to_numpy()[:, 1:4]
    x = xset.to_numpy()[:, 1:3]
    success, r, t, inlierIdx = cv2.solvePnPRansac(X, x, K, np.zeros((4, 1)))
    #print(inlierIdx)
    print(r)
    # r = r / LA.norm(r)
    # theta = LA.norm(r)
    # R = np.eye(3) + np.sin(theta) * np.cross(np.eye(3), r) + (1 - np.cos(theta)) * np.dot(r, r.T)
    R, jacobian = cv2.Rodrigues(r)
    print(R)
    print(t)
    Inlier = Xset.loc[inlierIdx[0]]
    '''
        
    # Decompose Pnew to obtain rotation R and camera center C
    R = P_new[:, 0:3]
    t = P_new[:, 3]
    #print(R)
    #print(t)
    '''

    Cnew = LA.inv(-R.T) @ t
    
    # Enforce orthogonality of R
    U, D, Vh = LA.svd(R)
    Rnew = U @ Vh

    if LA.det(Rnew) < 0:
        Rnew = -Rnew
    
    return Cnew, Rnew, Inlier  # Return the estimated camera center, rotation matrix, and inliers
