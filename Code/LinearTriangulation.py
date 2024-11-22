# import numpy as np
# import sys
# from numpy import linalg as LA
#
#
# # Function to perform Linear Triangulation to estimate 3D points from two camera views
# def LinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set):
#     """
#     LinearTriangulation: Computes 3D points from two sets of 2D correspondences (x1set and x2set)
#     observed from two different camera poses using linear triangulation.
#
#     Parameters:
#     - K: Intrinsic camera matrix (3x3).
#     - C0: Camera center for the first camera pose (3x1).
#     - R0: Rotation matrix for the first camera pose (3x3).
#     - Cseti: Camera center for the second camera pose (3x1).
#     - Rseti: Rotation matrix for the second camera pose (3x3).
#     - x1set: DataFrame containing 2D points in the first image (ID, u, v).
#     - x2set: DataFrame containing corresponding 2D points in the second image (ID, u, v).
#
#     Returns:
#     - Xset: Array of 3D points in homogeneous coordinates along with their IDs (Nx4).
#     """
#     x1set = x1set.to_numpy()
#     x2set = x2set.to_numpy()
#
#     I = np.identity(3)
#     C1 = np.reshape(C0, (3, 1))
#     C2 = np.reshape(Cseti, (3, 1))
#
#     P1 = np.dot(K, np.dot(R0, np.hstack((I, -C1))))
#     P2 = np.dot(K, np.dot(Rseti, np.hstack((I, -C2))))
#
#     p1T = P1[0, :].reshape(1, 4)
#     p2T = P1[1, :].reshape(1, 4)
#     p3T = P1[2, :].reshape(1, 4)
#
#     p_dash_1T = P2[0, :].reshape(1, 4)
#     p_dash_2T = P2[1, :].reshape(1, 4)
#     p_dash_3T = P2[2, :].reshape(1, 4)
#
#     X_set = []
#     for i in range(x1set.shape[0]):
#         ID = x1set[i,0]
#         x = x1set[i, 1]
#         y = x1set[i, 2]
#         x_dash = x2set[i, 1]
#         y_dash = x2set[i, 2]
#
#         A = []
#         A.append((y * p3T) - p2T)
#         A.append(p1T - (x * p3T))
#         A.append((y_dash * p_dash_3T) - p_dash_2T)
#         A.append(p_dash_1T - (x_dash * p_dash_3T))
#
#         A = np.array(A).reshape(4, 4)
#
#         _, _, vt = np.linalg.svd(A)
#         v = vt.T
#         x = v[:, -1]
#         X_set.append([ID, x[0], x[1], x[2]])
#
#     return X_set  # Return the set of triangulated 3D points with their IDs

import numpy as np
import sys
from numpy import linalg as LA

#Function to perform Linear Triangulation to estimate 3D points from two camera views
def LinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set):
    """
    LinearTriangulation: Computes 3D points from two sets of 2D correspondences (x1set and x2set)
    observed from two different camera poses using linear triangulation.

    Parameters:
    - K: Intrinsic camera matrix (3x3).
    - C0: Camera center for the first camera pose (3x1).
    - R0: Rotation matrix for the first camera pose (3x3).
    - Cseti: Camera center for the second camera pose (3x1).
    - Rseti: Rotation matrix for the second camera pose (3x3).
    - x1set: DataFrame containing 2D points in the first image (ID, u, v).
    - x2set: DataFrame containing corresponding 2D points in the second image (ID, u, v).

    Returns:
    - Xset: Array of 3D points in homogeneous coordinates along with their IDs (Nx4).
    """

    Xset = []  # List to store the triangulated 3D points

    # print('x1 set before numpy conversion')
    # print(x1set)
    # print('x2 set before numpy conversion')
    # print(x2set)
    # # Convert DataFrames to numpy arrays for easier manipulation
    x1set = x1set.to_numpy()
    x2set = x2set.to_numpy()
    print('Testing starts here: \n\n\n')
    print('x1\n')
    print(x1set)
    print('x2\n')
    print(x2set)

    print('r0\n')
    print(R0)
    print('Rseti\n')
    print(Rseti)
    print('C0\n')
    print(C0)
    print('Cseti \n')
    print(Cseti)
    print('K\n')
    print(K)

    # Calculate the projection matrices P1 and P2 with the form P = KR[I|-C]
    I = np.eye(3)  # Identity matrix (3x3)
    P1 = np.matmul(np.matmul(K, R0), np.concatenate((I, -C0), axis=1))  # Projection matrix for the first camera
    P2 = np.matmul(np.matmul(K, Rseti), np.concatenate((I, (-1) * Cseti), axis=1))  # Projection matrix for the second camera

    # Iterate over each pair of corresponding points (x1 in first view and x2 in second view)
    for x1, x2 in zip(x1set, x2set):
        ID = x1[0]  # Unique ID for the point
        u1 = x1[1]  # x-coordinate in the first image
        v1 = x1[2]  # y-coordinate in the first image
        w1 = 1

        u2 = x2[1]  # x-coordinate in the second image
        v2 = x2[2]  # y-coordinate in the second image
        w2 = 1

        # Construct matrix A for the linear triangulation system Ax=0
        # Each row in A is derived from the epipolar geometry constraint
        Chi_1 = np.array([[0, -w1, v1],[w1, 0, -u1],[-v1, u1, 0]])
        Chi_2 = np.array([[0, -w2, v2], [w2, 0, -u2], [-v2, u2, 0]])

        #A = np.vstack( (Chi_1 @ P1, Chi_2 @ P2))
        A = np.array([
            u1*P1[2,:] - P1[0,:],
            v1*P1[2,:] - P1[1,:],
            u2*P2[2,:] - P2[0,:],
            v2*P2[2,:] - P2[1,:]
        ])

        # A = np.array([
        #     x1[0] * P_1[2, :] - P_1[0, :],
        #     x1[1] * P_1[2, :] - P_1[1, :],
        #     x2[0] * P_2[2, :] - P_2[0, :],
        #     x2[1] * P_2[2, :] - P_2[1, :]
        # ])
        # Solve Ax=0 using the eigenvector associated with the smallest eigenvalue of A^T A
        # Compute A^T * A
        # Eigen decomposition of A^T * A
        U, S, Vh = LA.svd(A.T @ A)

        # Find the smallest eigenvalue

        # Corresponding eigenvector gives the solution
        # Normalize to make the point homogeneous
        X_i = Vh[-1] / Vh[-1,-1]

        # Append the triangulated 3D point with its ID to the list
        Xset.append([ID, X_i[0], X_i[1],X_i[2]])

    # Convert the list of points to a numpy array for easy manipulation
    Xset = np.array(Xset)
    print('This is the X_set:\n')
    print(Xset)
    return Xset  # Return the set of triangulated 3D points with their IDs

#
# import numpy as np
# import cv2
#
#
# def point_triangulation(k, pt1, pt2, R1, C1, R2, C2):
#     Xset = []
#
#     I = np.identity(3)
#     C1 = C1.reshape(3, 1)
#     print(C2)
#     C2 = C2.reshape(3, 1)
#
#     # calculating projection matrix P = K[R|T]
#     P1 = np.dot(k, np.dot(R1, np.hstack((I, -C1))))
#     P2 = np.dot(k, np.dot(R2, np.hstack((I, -C2))))
#
#     # homogeneous coordinates for images
#     xy = np.hstack((pt1, np.ones((len(pt1), 1))))
#     xy_cap = np.hstack((pt2, np.ones((len(pt1), 1))))
#
#     p1, p2, p3 = P1
#     p1_cap, p2_cap, p3_cap = P2
#
#     # constructing contraints matrix
#     for i in range(len(xy)):
#         A = []
#         ID = xy[i][0]
#         x = xy[i][1]
#         y = xy[i][2]
#         x_cap = xy_cap[i][1]
#         y_cap = xy_cap[i][2]
#
#         A.append((y * p3) - p2)
#         A.append((x * p3) - p1)
#
#         A.append((y_cap * p3_cap) - p2_cap)
#         A.append((x_cap * p3_cap) - p1_cap)
#
#         A = np.array(A).reshape(4, 4)
#
#         _, _, v = np.linalg.svd(A)
#         x_ = v[-1, :]
#         x_ = x_ / x_[-1]
#         # x_ =x_[:3]
#         Xset.append([ID, x_[0], x_[1], x_[2]])
#
#     Xset = np.array(Xset)
#     print('This is the X_set:\n')
#     print(Xset)
#     return Xset
#
#
# def LinearTriangulation(k, C1_, R1_, T_Set, R_Set, pt1, pt2):
#     R1_ = np.identity(3)
#     T1_ = np.zeros((3, 1))
#
#     # pt1 = pt1.to_numpy()
#     # pt2 = pt2.to_numpy()
#     points_3d_set = []
#
#
#
#     points_3d_set = point_triangulation(k, pt1, pt2, R1_, T1_, R_Set, T_Set)
#
#
#     return points_3d_set
#
