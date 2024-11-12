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

    # RANSAC initialization
    inliersMax = 0
    FBest = None
    x1Inlier, x2Inlier = None, None

    for i in tqdm(range(M)):
        # Step 1: Randomly select 8 pairs of points
        randIdxs = np.random.choice(x1All.index, size=8, replace=False)
        x1 = x1All.loc[randIdxs, [2, 3]]
        x2 = x2All.loc[randIdxs, [5, 6]]

        # Step 2: Estimate Fundamental matrix F
        F = EstimateFundamentalMatrix(x1, x2)

        # Step 3: Check each point pair to see if it satisfies the epipolar constraint
        inliersIdx = []

        for j in x1All.index:
            # Accessing x and y coordinates for source (x1All) and target (x2All) keypoints
            m1 = np.append(x1All.loc[j, [2, 3]].to_numpy(), 1)  # For x1All, columns 2 and 3 (indices 1 and 2)
            m2 = np.append(x2All.loc[j, [5, 6]].to_numpy(), 1)  # For x2All, columns 5 and 6 (indices 5 and 6)

            # Calculate the epipolar constraint error
            error = np.abs(np.dot(m2.T, np.dot(F, m1)))

            if error < T:
                inliersIdx.append(j)

        # Step 4: Update the best F if current F has more inliers
        if len(inliersIdx) > inliersMax:
            inliersMax = len(inliersIdx)
            FBest = F
            x1Inlier = x1All.loc[inliersIdx]
            x2Inlier = x2All.loc[inliersIdx]
    print(inliersMax)
    return x1Inlier, x2Inlier, FBest


