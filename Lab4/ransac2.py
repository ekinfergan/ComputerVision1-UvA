import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()
import ransac


def ransac(kp1, kp2, matches, N):
    """
      Apply RANSAC to filter out the outliers. See the pseudocode provided at
      the beginning of this section for some guidance.
      Arguments:
        kp1: the keypoints of img1
        kp2: the keypoints of img2
        matches: the matching kp1 and kp2
        N: number of iterations
      Returns:
        the best transformation matrix
    """

    # ================
    # OUR CODE HERE
    # ================
    num_best_inliers = 0
    best_matches = []
    best_matrix = np.zeros((3, 3))
    P = 10
    num_matches = len(matches)
    for i in range(N):
        print(f"\nItr No:{i + 1}")
        index = np.random.randint(0, num_matches, P)
        A = []
        b = []
        for j in index:
            # Get the indexes
            match = matches[j]
            img1_idx = match.queryIdx  # Index of the keypoint in the first image
            img2_idx = match.trainIdx  # Index of the keypoint in the second image
            sub_f_image1 = kp1[img1_idx]
            sub_f_image2 = kp2[img2_idx]
            x_img1, y_img1 = sub_f_image1.pt
            x_img2, y_img2 = sub_f_image2.pt
            A.append([x_img1, y_img1, 0, 0, 1, 0])
            A.append([0, 0, x_img1, y_img1, 0, 1])
            b.extend([x_img2, y_img2])

        A = np.array(A)
        b = np.array(b)
        b = b.reshape(-1, 1)
        A_p = np.linalg.pinv(A)
        affine_params = np.dot(A_p, b)
        m1, m2, m3, m4, t1, t2 = affine_params.ravel()
        affine_params = np.array([[m1, m2, t1], [m3, m4, t2]])

        matches_pairs = []
        num_inliers = 0
        for pair in matches:
            x1, y1 = kp1[pair.queryIdx].pt
            x2, y2 = kp2[pair.trainIdx].pt
            homo_x1 = np.array([x1, y1, 1])
            estimate_point2 = affine_params @ homo_x1
            all_p2 = np.array([x2, y2])
            errors = np.linalg.norm(all_p2 - estimate_point2)
            if errors < 10:
                num_inliers += 1
                matches_pairs.append([x1, y1, estimate_point2[0], estimate_point2[1]])
        if num_inliers > num_best_inliers:
            num_best_inliers = num_inliers
            best_matrix = affine_params.copy()
            best_matches = matches_pairs

    print("Total number of matches: ", len(matches))
    print("Inliers found:           ", num_best_inliers)
    print("Outliers removed:        ", len(matches) - num_best_inliers)

    return best_matrix, np.array(best_matches)



img1_path = "images/sp1.jpg"
img2_path = "images/sp2.jpg"

# # Open images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Note: OpenCV uses BGR instead of RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

keypoints_1, keypoints_2, matches = ransac.keypoint_matching(img1,img2)
best_matrix = ransac.ransac(keypoints_1, keypoints_2, matches, 10)
ransac.visualization(img1, img2, best_matrix)