import pandas as pd
import numpy as np
import random
import sys

from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from tqdm import tqdm

def GetInliersRANSAC(x1All, x2All, M=1500, T=0.5):
    """
    Estimates the Fundamental matrix using RANSAC and identifies inlier matches
    between two sets of points, rejecting outliers.

    Args:
        x1All (DataFrame): Source image points with IDs and (x, y) coordinates.
        x2All (DataFrame): Target image points with IDs and (x, y) coordinates.
        M (int): Number of RANSAC iterations. Default is 1500.
        T (float): Threshold for inlier selection based on the epipolar constraint. Default is 0.5.

    Returns:
        x1Inlier (DataFrame): Inlier points in the source image.
        x2Inlier (DataFrame): Inlier points in the target image.
        FBest (ndarray): The best estimated Fundamental matrix.
    """
 
    # RANSAC iterations
    inliersMax = 0
    
    for i in tqdm(range(M)):
        # Step 1: Randomly select 8 pairs of points from the source and target images
        # Extract coordinates without IDs for Fundamental matrix estimation
        randIdxs = np.random.randint(x1All.shape[0], size = 8)
        x1 = x1All.loc[randIdxs]
        x2 = x2All.loc[randIdxs]
        x1 = x1.to_numpy()
        x2 = x2.to_numpy()
        x1 = x1[:, 1:]
        x2 = x2[:, 1:]
        
        # Step 2: Estimate the Fundamental matrix F from the selected 8-point subsets
        # Call EstimateFundamentalMatrix function here.
        F = EstimateFundamentalMatrix(x1, x2)

        # Step 3: Check each point pair to see if it satisfies the epipolar constraint
        inliersIdx = np.array([])
        max_idx = x1All.shape[0]

        for j in range(max_idx): # max_idx: Total number of points
            # Calculate the epipolar constraint error for the source-target pair
            m1 = (x1All.loc[j]).to_numpy()[1:]
            m2 = (x2All.loc[j]).to_numpy()[1:]
            m1 = np.append(m1, 1)
            m2 = np.append(m2, 1)

            dist = np.sqrt(np.sum((m2 @ F @ m1.T) ** 2))

            # If the epipolar constraint error is below the threshold T, consider it an inlier
            if dist <= T:
                inliersIdx = np.append(inliersIdx, j)

        # Step 4: Update the best Fundamental matrix if the current F has more inliers
        if inliersIdx.size > inliersMax:
            FBest = F
            x1Inlier = x1All.loc[inliersIdx]
            x2Inlier = x2All.loc[inliersIdx]

    # Return the inlier sets and the best Fundamental matrix
    return x1Inlier, x2Inlier, FBest
