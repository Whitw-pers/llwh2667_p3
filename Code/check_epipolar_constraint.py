import pandas as pd
import numpy as np

def check_epipolar_constraint_with_print(F, source_inliers, target_inliers, threshold=1e-4):
    """
    Checks if the fundamental matrix F satisfies the epipolar constraint for corresponding points
    and prints the constraint values for each point pair.

    Args:
        F (ndarray): The fundamental matrix.
        source_inliers (DataFrame): Inlier points from the first image (image 1).
        target_inliers (DataFrame): Inlier points from the second image (image 2).
        threshold (float): The threshold below which the constraint should be satisfied.

    Returns:
        None
    """
    # Extract points from the source and target inliers by using correct column labels
    points1 = source_inliers.iloc[:, [1, 2]].values  # Columns 1 and 2 (x, y) from source image (image 1)
    points2 = target_inliers.iloc[:, [1, 2]].values  # Columns 1 and 2 (x, y) from target image (image 2)

    # Convert points to homogeneous coordinates (append 1)
    points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])  # Shape (N, 3)
    points2_h = np.hstack([points2, np.ones((points2.shape[0], 1))])  # Shape (N, 3)

    # Iterate over the points and compute the epipolar constraint
    for i in range(points1.shape[0]):
        x1 = points1_h[i, :].reshape(3, 1)  # Column vector of point in image 1 (homogeneous coordinates)
        x2 = points2_h[i, :].reshape(3, 1)  # Column vector of point in image 2 (homogeneous coordinates)

        # Compute the epipolar constraint: x2^T * F * x1
        constraint = float(x2.T @ F @ x1)

        # Print the constraint value for each point pair
        print(f"Point pair {i + 1}: x2^T F x1 = {constraint}")

        # Check if the constraint is satisfied
        if abs(constraint) > threshold:
            print(f"Constraint violation for point pair {i + 1}: {constraint}")

# Example usage:
# Assuming source_inliers and target_inliers are pandas DataFrames containing inlier points

fundamental_matrix_test = np.array([
    [-4.57648790e-06,  8.22281981e-05, -1.57269056e-02],
    [-6.71109825e-05, -7.98418684e-06,  2.38994659e-02],
    [ 2.53186146e-02, -1.33985162e-02, -5.57525961e+00]
])

# Data: [ID, x-coordinate, y-coordinate]
data_source = [
    [1, 1002.25, 206.56],
    [2, 1002.59, 218.98],
    [3, 1002.65, 218.67],
    [5, 1002.74, 211.13],
    [7, 1004.26, 451.88],
    [10, 1005.73, 233.93],
    [17, 1011.24, 204.57],
    [18, 1012.88, 224.27],
    [19, 1012.94, 497.04],
    [20, 1014.77, 155.31],
    [27, 1020.39, 446.53],
    [28, 1020.89, 398.88],
    [31, 1021.81, 500.19],
    [32, 1022.83, 212.49],
    [34, 1023.95, 279.05],
    [35, 1024.09, 430.17],
    [36, 1024.10, 225.60],
    [37, 1024.37, 248.65],
    [38, 1024.81, 263.34],
    [39, 1027.16, 179.24],
    [41, 1027.89, 179.83],
    [43, 1028.27, 453.06],
    [45, 1029.39, 407.25],
    [46, 1029.68, 393.49],
    [49, 1030.58, 508.23],
    [50, 1031.60, 218.70],
    [51, 1031.64, 233.45],
    [53, 1035.84, 515.43],
    [55, 1037.57, 236.98],
    [56, 1038.10, 426.30],
    [58, 1039.29, 312.45],
    [59, 1040.25, 444.88],
    [61, 1040.65, 241.57],
    [62, 1041.31, 234.96],
    [63, 1041.63, 218.31],
    [64, 1042.04, 352.27],
    [66, 1042.81, 469.25],
    [67, 1043.08, 481.14],
    [69, 1044.26, 252.20],
    [70, 1045.30, 231.47],
    [71, 1048.00, 271.76],
    [72, 1048.29, 406.66],
    [73, 1049.87, 174.19],
    [75, 1050.00, 394.96],
    [76, 1050.03, 222.85],
    [78, 1052.70, 235.13],
    [79, 1053.14, 231.60],
    [81, 1055.64, 402.09],
    [85, 1059.02, 264.69],
    [86, 1059.09, 364.15]
]

import pandas as pd

# Create the data
data_target = [
    [1, 862.75, 329.44],
    [2, 862.34, 338.35],
    [3, 862.34, 338.35],
    [5, 862.04, 332.52],
    [7, 889.89, 547.06],
    [10, 866.37, 351.71],
    [16, 867.75, 327.54],
    [17, 870.31, 343.66],
    [18, 901.37, 590.88],
    [19, 864.75, 286.76],
    [25, 901.19, 540.85],
    [26, 896.60, 498.74],
    [29, 908.79, 593.17],
    [30, 876.50, 334.76],
    [32, 883.95, 388.29],
    [33, 904.74, 525.17],
    [34, 878.55, 345.68],
    [35, 881.45, 365.27],
    [36, 883.71, 377.65],
    [37, 875.56, 306.98],
    [39, 875.56, 306.98],
    [41, 908.14, 547.82],
    [43, 903.39, 505.02],
    [44, 902.58, 492.63],
    [47, 915.91, 600.58],
    [48, 883.45, 339.76],
    [49, 884.72, 353.14],
    [50, 920.65, 608.47],
    [52, 889.60, 355.70],
    [53, 911.99, 522.95],
    [55, 897.80, 419.80],
    [56, 915.86, 538.32],
    [58, 892.52, 359.84],
    [59, 892.54, 354.05],
    [60, 890.60, 339.41],
    [61, 905.48, 456.02],
    [63, 920.64, 562.12],
    [64, 922.04, 572.62],
    [66, 896.38, 368.06],
    [67, 894.39, 350.68],
    [68, 900.04, 384.54],
    [69, 917.02, 503.81],
    [70, 891.75, 304.63],
    [71, 917.01, 494.06],
    [72, 897.49, 344.35],
    [74, 899.75, 354.58],
    [75, 900.07, 351.58],
    [77, 921.70, 498.17],
    [81, 906.57, 379.80],
    [82, 845.70, 456.10]
]


# Create DataFrame
source_inliers = pd.DataFrame(data_source, columns=[0, 1, 2])
target_inliers = pd.DataFrame(data_target, columns=[0 , 1, 2])

# Call the function to check the epipolar constraint
check_epipolar_constraint_with_print(fundamental_matrix_test, source_inliers, target_inliers)
