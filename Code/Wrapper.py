"""
confirm cwd is /Code to run successfully

This is a startup script to processes a set of images to perform Structure from Motion (SfM) by
extracting feature correspondences, estimating camera poses, and triangulating 3D points, 
performing PnP and Bundle Adjustment.
"""
from Code.BundleAdjustment import BundleAdjustment
from NonlinearTriangulation import NonlinearTriangulation
#from Code.NonlinearTriangulation import NonlinearTriangulation
from PlotPtsCams import PlotPtsCams
#from Code.PlotPtsCams import PlotPtsCams
from TriangulationComp import TriangulationComp
#from Code.TriangulationComp import TriangulationComp

'''
from Code.DisambiguateCameraPose import DisambiguateCameraPose
from Code.EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from Code.ExtractCameraPose import ExtractCameraPose
from Code.LinearTriangulation import LinearTriangulation
from Code.PlotCameraPts import PlotCameraPts
from Code.utils import DrawMatches
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from utils import IndexAllFeaturePoints, process_calib_matrix
from utils import ParseKeypoints
import os
from utils import bcolors
import cv2
from GetInliersRANSAC import GetInliersRANSAC
from Draw_Epipolar_Lines import Draw_Epipolar_Lines
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
'''

from utils import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from PnPRANSAC import *
#from NonlinearTriangulation import NonlinearTriangulation
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import *
#from BundleAdjustment import BundleAdjustment
from Draw_Epipolar_Lines import *
from PlotCameraPts import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

################################################################################
# Step 1: Parse all matching files and assign IDs to feature points.
# Each file contains matches between feature points in successive images.
################################################################################

file1 = '../Data/Imgs/matching1.txt'
file2 = '../Data/Imgs/matching2.txt'
file3 = '../Data/Imgs/matching3.txt'
file4 = '../Data/Imgs/matching4.txt'
file5 = '../Data/Imgs/matching5.txt'

"""
Assign Unique IDs to feature points across datasets.
"""

"""
The IndexAllFeaturePoints function takes five text files (file1 through file5), each representing feature point matches from different images. It processes each file to:
1. Extract and clean feature point data.
2. Assign unique identifiers (IDs) to each feature point.
3. Ensure that feature points shared across different files are assigned the same ID.
"""

# Check if processed matching files already exist to avoid redundant processing.
if not os.path.exists('../Data/new_matching1.txt'):
    print("\nProcessing Feature Correspondences from matching files...")
    # Process the matching files to assign a unique ID to each feature point.
    # This enables tracking of each point across multiple images in the dataset.
    # Each feature point across the entire dataset will have a unique index.
    match1DF, match2DF, match3DF, match4DF, match5DF = IndexAllFeaturePoints(file1, file2, file3, file4, file5)

else:
    print('Refined Features Indexes Already Exists')
    # Refer utils.py for color definiton
    print(bcolors.WARNING + "Warning: Continuing with the existing Feature Indexes..." + bcolors.ENDC)


################################################################################
# Step 2: Parse matching file for the first pair of cameras (cameras 1 and 2).
# Each file contains matches between feature points in successive images.
################################################################################

# Define the file path and camera indices for parsing keypoints.

file_path = '../Data/new_matching1.txt'
source_camera_index = 1
target_camera_index = 2


# Execute the keypoint parsing for the specified camera pair.
# The output DataFrame provides a structured set of matched keypoints between two images.
ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)

# Extract coordinates for source and target keypoints
# Select specific columns to represent keypoint coordinates in each image
# - 0: Keypoint ID, which uniquely identifies each match
# - 2, 3: X and Y coordinates of the keypoint in the source image
# - 5, 6: X and Y coordinates of the corresponding keypoint in the target image
source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

# print(source_keypoints)
# print(target_keypoints)

################################################################################
# Step 3: RANSAC-based outlier rejection.
# Remove outlier based on fundamental matrix. For every eight points, you can
# get one fundamental matrix. But only one fundamental matrix is correct between
# two images. Optimize and find the best fundamental matrix and remove outliers
# based on the features that corresponds to that fundamental matrix.
# This step refines the matches by removing outliers and retaining inliers.
################################################################################

# Use RANSAC to estimate a robust Fundamental matrix (F) that minimizes the impact of outliers.
# Write a function GetInliersRANSAC that removes outliers and compute Fundamental Matrix
# using initial feature correspondences

ransac_output_file = '../Data/ransac_inliers.pkl'

if os.path.exists(ransac_output_file):
    # Load saved RANSAC results
    print("Loading saved RANSAC results...")
    with open(ransac_output_file, 'rb') as f:
        source_inliers, target_inliers, fundamental_matrix = pickle.load(f)
else:
    # Run GetInliersRANSAC and save the results
    print("Running RANSAC for the first time...")
    source_inliers, target_inliers, fundamental_matrix = GetInliersRANSAC(source_keypoints, target_keypoints)

    # Save the results to a file
    with open(ransac_output_file, 'wb') as f:
        pickle.dump((source_inliers, target_inliers, fundamental_matrix), f)
    print("RANSAC results saved.")

source_coord = source_inliers.loc[:, [2, 3]]  # Select all rows, columns 2 and 3 (x and y coordinates for source points)
target_coord = target_inliers.loc[:, [5, 6]]  # Select all rows, columns 5 and 6 (x and y coordinates for target points)

fundamental_matrix_refined = EstimateFundamentalMatrix(source_coord,target_coord)

image1 = cv2.imread('../Data/Imgs/1.jpg')
image2 = cv2.imread('../Data/Imgs/2.jpg')

Draw_Epipolar_Lines(image1, image2, source_inliers, target_inliers, fundamental_matrix_refined, num_inliers=20)

#################################################################################
# You will write another function 'EstimateFundamentalMatrix' that computes F matrix
# This function is being called by the 'GetInliersRANSAC' function
#################################################################################



# Visualize the final feature correspondences after computing the correct Fundamental Matrix.
# Write a code to print the final feature matches and compare them with the original ones.
# Assuming source_keypoints, target_keypoints, source_inliers, target_inliers are your DataFrames
source_keypoints_array = source_keypoints.iloc[:, -2:].values  # Last two columns for source
target_keypoints_array = target_keypoints.iloc[:, -2:].values  # Last two columns for target
source_inliers_array = source_inliers.iloc[:, -2:].values  # Last two columns for source inliers
target_inliers_array = target_inliers.iloc[:, -2:].values  # Last two columns for target inliers

print('Initial Source Points\n')
print(source_inliers)
print('Initial Target Points\n')
print(target_inliers)
# Now pass these arrays to the DrawMatches function
DrawMatches('../Data/Imgs', 1, 2, source_keypoints_array, target_keypoints_array, source_inliers_array, target_inliers_array, '../Data/Imgs/RansacTester')
################################################################################
# Step 4: Load intrinsic camera matrix K, which contains focal lengths and 
# principal point.
# The calibration file provides intrinsic parameters which are used to calculate
# the Essential matrix.
################################################################################

calib_file = '../Data/Imgs/calibration.txt'
K = process_calib_matrix(calib_file)
print(bcolors.OKCYAN + "\nIntrinsic camera matrix K:" + bcolors.OKCYAN)
print(K, '\n')


################################################################################
# Step 5: Compute Essential Matrix from Fundamental Matrix
################################################################################
E = EssentialMatrixFromFundamentalMatrix(fundamental_matrix_refined, K)
print(bcolors.OKGREEN + "\nEssential camera matrix E:" + bcolors.OKGREEN)
print(E, '\n')

################################################################################
# Step 6: Extract Camera Poses from Essential Matrix
# Note: You will obtain a set of 4 translation and rotation from this function
################################################################################
Cset, Rset = ExtractCameraPose(E)
print(bcolors.OKGREEN + "\nPotential camera Poses:" + bcolors.OKGREEN)
print(Cset, '\n')
print(Rset, '\n')


################################################################################
# Step 6: Linear Triangulation
################################################################################
# Initialize an empty list to store the 3D points calculated for each camera pose
Xset = []
# Iterate over each camera pose in Cset and Rset
for i in range(4):
    # Perform linear triangulation to estimate the 3D points given:
    # - K: Intrinsic camera matrix
    # - np.zeros((3,1)): The initial camera center (assumes origin for the first camera)
    # - np.eye(3): The initial camera rotation (identity matrix, assuming no rotation for the first camera)
    # - Cset[i]: Camera center for the i-th pose
    # - Rset[i]: Rotation matrix for the i-th pose
    # - x1Inlier: Inlier points in the source image
    # - x2Inlier: Corresponding inlier points in the target image
    Xset_i = LinearTriangulation(K, np.zeros((3,1)), np.eye(3), Cset[i], Rset[i], source_inliers, target_inliers)

    Xset.append(Xset_i)


################################################################################
## Step 7: Plot all points and camera poses
# Write a function: PlotPtsCams that visualizes the 3D points and the estimated camera poses.
################################################################################

# Arguments:
# - Cset: List of camera centers for each pose.
# - Rset: List of rotation matrices for each pose.
# - Xset: List of triangulated 3D points corresponding to each camera pose.
# - SAVE_DIR: Output directory to save the plot.
# - FourCameraPose.png: Filename for the output plot showing 3D points and camera poses.

SAVE_DIR = '../Data/Imgs'
PlotCameraPts(Cset, Rset, Xset, SAVE_DIR, 'FourCameraPose.png')

print('first point pose 1\n')
print(Xset[0])
print('first point pose 2\n')
print(Xset[1])
print('first point pose 3\n')
print(Xset[2])
print('first point pose 4\n')
print(Xset[3])

################################################################################
## Step 8: Disambiguate Camera Pose
# Write a function: DisambiguateCameraPose
# DisambiguateCameraPose is called to identify the correct camera pose from multiple
# hypothesized poses. It selects the pose with the most inliers in front of both 
# cameras (i.e., the pose with the most consistent triangulated 3D points).
################################################################################

## Disambiguate camera poses
# Arguments:
# - Cset: List of candidate camera centers for each pose.
# - Rset: List of candidate rotation matrices for each pose.
# - Xset: List of sets of triangulated 3D points for each camera pose.
# Returns:
# - C: The selected camera center after disambiguation.
# - R: The selected rotation matrix after disambiguation.
# - X: The triangulated 3D points for the selected camera pose.
# - selectedIdx: The index of the selected camera pose within the original candidate sets.
C, R, X, selectedIdx = DisambiguateCameraPose(Cset, Rset, Xset)

print(f"Correct Camera Pose Index: {selectedIdx}")
print(f"Camera Center: \n{C}")
print(f"Rotation Matrix: \n{R}")
print(f"Corresponding 3D Points: \n{X}")

# Plot the selected camera pose with its 3D points
# This plot shows the selected camera center, orientation, and the corresponding 3D points.
# Arguments:
# - [C]: List containing the selected camera center (wrapping it in a list for compatibility with PlotPtsCams).
# - [R]: List containing the selected rotation matrix.
# - [X]: List containing the 3D points for the selected camera pose.
# - SAVE_DIR: Output directory to save the plot.
# - OneCameraPoseWithPoints.png: Filename for the output plot showing both the camera pose and 3D points.
# - show_pos=True: Enables the display of the camera pose.
PlotPtsCams([C], [R], [X], SAVE_DIR, 'OneCameraPoseWithPoints.png')

################################################################################
## Step 9: Non-Linear Triangulation
# Write a function: NonLinearTriangulation
# Inputs:
# - K: Intrinsic camera matrix of the first camera (3x3).
# - C0, R0: Translation (3x1) and rotation (3x3) of the first camera.
# - Cseti, Rseti: Translations and rotations of other cameras in a list.
# - x1set, x2set: Sets of 2D points in each image for triangulation.
# - X0: Initial 3D points for optimization.
# Output:
# - Returns optimized 3D points after minimizing reprojection error.
XOPT_FILE = os.path.join(SAVE_DIR, "Xopt.npy")

if os.path.exists(XOPT_FILE):
    # Load Xopt if it exists
    print("Loading precomputed Xopt from file...")
    Xopt = np.load(XOPT_FILE)
else:
    # Compute Xopt using NonlinearTriangulation if it doesn't exist
    print("Computing Xopt using NonlinearTriangulation...")
    Xopt = NonlinearTriangulation(K, np.zeros((3, 1)), np.eye(3), C, R, source_inliers, target_inliers, X)

    # Save the computed Xopt to a file
    np.save(XOPT_FILE, Xopt)
    print(f"Xopt saved to {XOPT_FILE}")

Cpair = [C, C]
Rpair = [R, R]
Xpair = [X, Xopt]
TriangulationComp(Cpair, Rpair, Xpair, SAVE_DIR, 'TriangulationComp.png')

################################################################################

################################################################################
# Step 10: PnPRANSAC
# PnPRANSAC: Function to perform PnP using RANSAC to find the best camera pose 
# with inliers
Xset = pd.DataFrame(Xopt)
xset = pd.DataFrame(source_inliers)

C1, R1, Inliers1 = PnPRANSAC(Xset, xset, K)
print(C)
print(C1)
print(R)
print(R1)
print(len(Inliers1))

#PlotPtsCams([C1], [R1], [Xset], SAVE_DIR, 'PnPRANSACresults.png')
Cpair = [C, C1]
Rpair = [R, R1]
Xpair = [Xopt, Xopt]
TriangulationComp(Cpair, Rpair, Xpair, SAVE_DIR, 'PnPRansacComp.png')

################################################################################

################################################################################
# Step 11: NonLinearPnP
# NonLinearPnP: Refines the camera pose (position and orientation) using non-linear 
# optimization to minimize the reprojection error between observed 2D points and 
# projected 3D points.

Copt1, Ropt1 = NonlinearPnP(Xset, xset, K, C1, R1)

Copt1 = Copt1.reshape((3, 1))

print(C1)
print(Copt1)
print(R1)
print(Ropt1)

Cpair = [C1, Copt1]
Rpair = [R1, Ropt1]
Xpair = [Xopt, Xopt]
TriangulationComp(Cpair, Rpair, Xpair, SAVE_DIR, 'NonlinearPnPComp.png')

################################################################################


################################################################################

print('This is what source inliers looks like:\n')
print(source_inliers)

print('This is what target inliers looks like:\n')
print(target_inliers)

print('This is Xopt\n')
print(Xopt)
source_inliers['CameraID'] = 0
target_inliers['CameraID'] = 1

# Extract PointID, u, v, CameraID
source_data = source_inliers[[0, 2, 3, 'CameraID']].to_numpy()
target_data = target_inliers[[0, 5, 6, 'CameraID']].to_numpy()

# Combine source and target inliers
combined_matrix = np.vstack((source_data, target_data))

xall = combined_matrix[:, 1:3].astype(float)
# Convert to DataFrame for better readability (optional)
combined_df = pd.DataFrame(combined_matrix, columns=['PointID', 'u', 'v', 'CameraID'])

print(combined_df)

camera_indices = combined_matrix[:, 3].astype(int)  # Camera IDs
point_indices = combined_matrix[:, 0].astype(int)  # Point IDs

camera_ids = combined_matrix[:, 3].astype(int)
n_cameras = np.unique(camera_ids).size
point_ids = combined_matrix[:, 0].astype(int)

# Find the number of unique points
n_points = np.unique(point_ids).size

visibility_matrix = BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices)
print(visibility_matrix)
# Step 12: BuildVisibilityMatrix
# BuildVisibilityMatrix: BuildVisibilityMatrix: Constructs a sparse visibility 
# matrix for bundle adjustment. This matrix indicates which parameters affect 
# each observation in the optimization process.
################################################################################

################################################################################
# Step 13: BundleAdjustment
# BundleAdjustment: Refines camera poses and 3D point positions to minimize the 
# reprojection error for a set of cameras and 3D points using non-linear 
# optimization.
################################################################################

# Wrapper function continues

# Iterate over remaining images (from the 3rd image onward)
# Initialize C_set and R_set as lists if not already done
C_set = []
R_set = []

# Then, within your loop:
C_set.append(Copt1)  # Appending the new camera center
R_set.append(Ropt1)  # Appending the new rotation matrix

n_cameras = 2  # Start with two cameras already considered
n_points = X.shape[0]  # Assuming X contains the initial set of 3D points

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)  # Create the directory if it does not exist

for i in range(2, 6):  # Adjust indices based on your image list structure
    file_path = '../Data/new_matching1.txt'
    source_camera_index = 1
    target_camera_index = i

    # Construct the filename for this iteration
    TRIANGULATION_FILE = os.path.join(SAVE_DIR, f"triangulation_results_{target_camera_index}.npy")

    # Execute the keypoint parsing for the specified camera pair.
    ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)
    source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
    target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

    # Properly rename columns to standardize
    source_keypoints.columns = ['FeatureID', 'u', 'v']  # Assuming columns 2 and 3 are 'u' and 'v'
    target_keypoints.columns = ['FeatureID', 'u', 'v']  # Assuming columns 5 and 6 are 'u' and 'v'

    source_keypoints.loc[:, 'FeatureID'] = source_keypoints['FeatureID'].astype(int)
    target_keypoints.loc[:, 'FeatureID'] = target_keypoints['FeatureID'].astype(int)

    # Convert Xopt to DataFrame if it's a numpy array
    if isinstance(Xopt, np.ndarray):
        Xopt = pd.DataFrame(Xopt, columns=['FeatureID', 'X', 'Y', 'Z'])

    # Merge target_keypoints with Xopt on 'FeatureID'
    matches = pd.merge(target_keypoints, Xopt, on='FeatureID')

    # Extract matched 3D points (Xset) and 2D points (xset)
    Xset = matches[['FeatureID', 'X', 'Y', 'Z']]  # Correct column names after merge
    xset = matches[['FeatureID', 'u', 'v']]  # Correct column names after merge

    # Continue with PnP RANSAC or other processing...

    # Step 2: Register the camera using PnP
    # Estimate the camera pose using PnPRANSAC
    C_new, R_new, inliers = PnPRANSAC(Xset, xset, K, M=2000, T=1000)


    # Refine the pose using NonlinearPnP
    C_new, R_new = NonlinearPnP(Xset, xset, K, C_new, R_new)
    C_new = C_new.reshape(-1, 1)  # Reshape C_new from (3,) to (3,1)
    # Step 3: Add the new camera pose to the set
    C_set.append(C_new)
    R_set.append(R_new)
    n_cameras += 1

    # Step 2: Triangulate new 3D points
    C0 = np.zeros((3, 1))  # Origin for the first camera
    R0 = np.eye(3)  # Identity rotation for the first camera

    matches_x1_x2 = pd.merge(source_keypoints, target_keypoints, on="FeatureID", suffixes=("_1", "_2"))

    x1set = matches_x1_x2[["FeatureID", "u_1", "v_1"]]
    x2set = matches_x1_x2[["FeatureID", "u_2", "v_2"]]

    if os.path.exists(TRIANGULATION_FILE):
        print(f"Loading precomputed triangulation results from {TRIANGULATION_FILE}...")
        refined_points_3d = np.load(TRIANGULATION_FILE)
    else:
        print("Computing triangulation results...")
        new_points_3d = LinearTriangulation(K, C0, R0, C_new, R_new, x1set, x2set)
        refined_points_3d = NonlinearTriangulation(K, C0, R0, C_new, R_new, x1set, x2set,new_points_3d)
        # Save the results
        np.save(TRIANGULATION_FILE, refined_points_3d)
        print(f"Triangulation results saved to {TRIANGULATION_FILE}")

    X = np.vstack((X, refined_points_3d))

    # Calculate the new indices starting from the last maximum index + 1
    if point_indices.size > 0:
        start_new_point_index = point_indices.max() + 1
    else:
        start_new_point_index = 0

    new_point_indices = np.arange(start_new_point_index, start_new_point_index + refined_points_3d.shape[0])
    new_camera_indices = np.full((refined_points_3d.shape[0],), i)

    # Concatenate new indices
    camera_indices = np.concatenate((camera_indices, new_camera_indices))
    point_indices = np.concatenate((point_indices, new_point_indices))

    # Update counts
    n_cameras = np.unique(camera_indices).size
    n_points = np.unique(point_indices).size + 1  # +1 as index starts at 0

    # Append new 2D points data to xall

    if isinstance(xset, pd.DataFrame) and 'u' in xset.columns and 'v' in xset.columns:
        new_xset_data = xset[['u', 'v']].to_numpy()  # Convert DataFrame to NumPy array
        xall = np.vstack([xall, new_xset_data])  # Stack the new data onto xall
    else:
        print("Error: xset does not contain 'u' and 'v' columns or is not properly formatted as a DataFrame.")

    # Build the visibility matrix for current state
    V = BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices)

    # Perform Bundle Adjustment (assuming C_set, R_set, K are defined and handled properly)
    C_opt, R_opt, X_opt = BundleAdjustment(C_set, R_set, X, K, V, n_cameras, n_points, camera_indices, point_indices,
                                           xall)

    # Optionally update the optimized values back to respective variables if needed for further iterations
    C_set = C_opt
    R_set = R_opt
    X = X_opt

# Save the entire set of camera settings and 3D points
with open('bundle_adjustment_data.pkl', 'wb') as f:
    pickle.dump({'C_set': C_set, 'R_set': R_set, 'X': X}, f)

# Load the data back
with open('bundle_adjustment_data.pkl', 'rb') as f:
    data = pickle.load(f)
    C_set = data['C_set']
    R_set = data['R_set']
    X = data['X']
