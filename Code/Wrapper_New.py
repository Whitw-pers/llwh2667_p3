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
        