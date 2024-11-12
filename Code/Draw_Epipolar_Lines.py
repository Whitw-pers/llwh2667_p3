import numpy as np
import matplotlib.pyplot as plt
import random


def Draw_Epipolar_Lines(image1, image2, inliers1, inliers2, F, num_inliers=20):
    """
    Draws epipolar lines on two images based on corresponding inliers and the fundamental matrix.

    Args:
        image1 (ndarray): The first image (source).
        image2 (ndarray): The second image (target).
        inliers1 (DataFrame): Inlier points from the first image.
        inliers2 (DataFrame): Inlier points from the second image.
        F (ndarray): The fundamental matrix.
        num_inliers (int): Number of inliers to plot epipolar lines for (default is 20).

    Returns:
        None
    """
    # Randomly select 'num_inliers' inliers for drawing the lines
    selected_indices = random.sample(range(len(inliers1)), num_inliers)
    selected_inliers1 = inliers1.iloc[selected_indices]
    selected_inliers2 = inliers2.iloc[selected_indices]

    # Create subplots to display the images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first image with epipolar lines
    ax1.imshow(image1)
    ax1.set_title("Image 1 with Epipolar Lines")
    ax1.axis('on')

    # Plot the second image with epipolar lines
    ax2.imshow(image2)
    ax2.set_title("Image 2 with Epipolar Lines")
    ax2.axis('on')

    # Draw epipolar lines on both images
    for i in range(num_inliers):
        # Get the corresponding points in the two images
        point1 = selected_inliers1.iloc[i][[2, 3]].values  # Use the 2nd and 3rd columns (x, y)
        point2 = selected_inliers2.iloc[i][[5, 6]].values  # Use the 5th and 6th columns (x, y)

        # Convert points to homogeneous coordinates
        point1_h = np.append(point1, 1)  # Homogeneous coordinates for point in image 1
        point2_h = np.append(point2, 1)  # Homogeneous coordinates for point in image 2

        # Compute the epipolar line in image 2 for the point in image 1
        line1 = F @ point1_h  # Epipolar line in image 2
        line2 = F.T @ point2_h  # Epipolar line in image 1

        # Get the coordinates for the line in image 2
        x1, y1 = 0, -line1[2] / line1[1]  # Solve for y when x = 0
        x2 = image2.shape[1]  # Max width of image 2
        y2 = -(line1[2] + line1[0] * x2) / line1[1]  # Solve for y when x = image2 width

        # Get the coordinates for the line in image 1
        x3, y3 = 0, -line2[2] / line2[1]  # Solve for y when x = 0
        x4 = image1.shape[1]  # Max width of image 1
        y4 = -(line2[2] + line2[0] * x4) / line2[1]  # Solve for y when x = image1 width

        # Plot line1 on image2 and line2 on image1
        ax2.plot([x1, x2], [y1, y2], color='r', linestyle='--', linewidth=1)  # Epipolar line in image 2
        ax1.plot([x3, x4], [y3, y4], color='r', linestyle='--', linewidth=1)  # Epipolar line in image 1

        # Plot the points on the images
        ax1.plot(point1[0], point1[1], 'bo')  # Blue point in image 1
        ax2.plot(point2[0], point2[1], 'bo')  # Blue point in image 2

    plt.tight_layout()
    plt.show()
