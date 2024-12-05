"""
confirm cwd is /Code to run successfully

This is a startup script to processes a set of images to perform Structure from Motion (SfM) by
extracting feature correspondences, estimating camera poses, and triangulating 3D points, 
performing PnP and Bundle Adjustment.
"""

from NonlinearTriangulation import NonlinearTriangulation
from PlotPtsCams import PlotPtsCams
from TriangulationComp import TriangulationComp
from utils import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from PnPRANSAC import *
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
# Step 2: For all possible pairs of images reject outlier correspondences
################################################################################

#file_path = '../Data/new_matching' + str(i + 1) + '.txt'
fp1 = '../Data/new_matching1.txt'
fp2 = '../Data/new_matching2.txt'
fp3 = '../Data/new_matching3.txt'
fp4 = '../Data/new_matching4.txt'
fp5 = '../Data/new_matching5.txt'

for i in range(5):
    for j in range(i + 1, 6):
        print("Reject outlier correspondences for images " + str(i + 1) + " and " + str(j + 1))

        # define file path and camera indices for parsing keypoints
        source_camera_index = i + 1
        target_camera_index = j + 1

        # Execute the keypoint parsing for the specified camera pair.
        # The output DataFrame provides a structured set of matched keypoints between two images.
        DF1 = ParseKeypoints(fp1, source_camera_index, target_camera_index)
        DF2 = ParseKeypoints(fp2, source_camera_index, target_camera_index)
        DF3 = ParseKeypoints(fp3, source_camera_index, target_camera_index)
        DF4 = ParseKeypoints(fp4, source_camera_index, target_camera_index)
        DF5 = ParseKeypoints(fp5, source_camera_index, target_camera_index)
        ParseKeypoints_DF = pd.concat([DF1, DF2, DF3, DF4, DF5], ignore_index=True)
        print(ParseKeypoints_DF)
        #ParseKeypoints_DF.to_csv('recent_data.csv', index = True)

        # Extract coordinates for source and target keypoints
        # Select specific columns to represent keypoint coordinates in each image
        # - 0: Keypoint ID, which uniquely identifies each match
        # - 2, 3: X and Y coordinates of the keypoint in the source image
        # - 5, 6: X and Y coordinates of the corresponding keypoint in the target image
        source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
        target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

        # Remove outlier based on fundamental matrix. For every eight points, you can
        # get one fundamental matrix. But only one fundamental matrix is correct between
        # two images. Optimize and find the best fundamental matrix and remove outliers
        # based on the features that corresponds to that fundamental matrix.
        # This step refines the matches by removing outliers and retaining inliers.
        
        ransac_output_file = '../Data/ransac_inliers'+ str(i+1) + str(j+1) + '.pkl'

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

################################################################################
# Step 3: For the first 2 images
################################################################################

# GET FUNDAMENTAL MATRIX (already calculated during RANSAC)
with open('../Data/ransac_inliers12.pkl', 'rb') as f:
    source_inliers12, target_inliers12, fundamental_matrix12 = pickle.load(f)
print(bcolors.OKGREEN + "\nFundamental matrix F:" + bcolors.OKGREEN)
print(fundamental_matrix12, '\n')

# GET ESSENTIAL MATRIX
# need to load camera intrisics K
calib_file = '../Data/Imgs/calibration.txt'
K = process_calib_matrix(calib_file)
print(bcolors.OKCYAN + "\nIntrinsic camera matrix K:" + bcolors.OKCYAN)
print(K, '\n')

E = EssentialMatrixFromFundamentalMatrix(fundamental_matrix12, K)
print(bcolors.OKGREEN + "\nEssential camera matrix E:" + bcolors.OKGREEN)
print(E, '\n')

# EXTRACT CAMERA POSE
Cset, Rset = ExtractCameraPose(E)
print(bcolors.OKGREEN + "\nPotential camera Poses:" + bcolors.OKGREEN)
print(Cset, '\n')
print(Rset, '\n')

# PERFORM LINEAR TRIANGULATION
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
    Xset_i = LinearTriangulation(K, np.zeros((3,1)), np.eye(3), Cset[i], Rset[i], source_inliers12, target_inliers12)

    Xset.append(Xset_i)

# ADD IN PLOTTING OF ALL CAMERA POSES
SAVE_DIR = '../Data/Imgs'
PlotCameraPts(Cset, Rset, Xset, SAVE_DIR, 'FourCameraPose.png')

# CHECK CHEIRALITY CONDITION TO DISAMBIGUATE CAMERA POSE
C, R, X, selectedIdx = DisambiguateCameraPose(Cset, Rset, Xset)

print(f"Correct Camera Pose Index: {selectedIdx}")
print(f"Camera Center: \n{C}")
print(f"Rotation Matrix: \n{R}")

# ADD IN PLOTTING OF SELECTED CAMERA POSE
PlotPtsCams([C], [R], [X], SAVE_DIR, 'OneCameraPoseWithPoints.png')

# PERFORM NONLINEAR TRIANGULATION'
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
# Step 4: For images 3-6 do
# estimate pose for the remaining cameras
################################################################################

for i in range(2, 6):
    ref_img = i
    new_img = i + 1

    # need to go through Xopt and find ids that correspond to those in the new_img

    