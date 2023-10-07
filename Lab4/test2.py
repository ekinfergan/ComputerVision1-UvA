import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import ransac
print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()


def stitch_ext(left, right, N=800):
    keypoints_1, keypoints_2, matches = ransac.keypoint_matching(left, right)
    best_ransac_matrix, _ = ransac.ransac(keypoints_1, keypoints_2, matches, N)
    best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)

    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(best_ransac_matrix_, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)
    x_min = -x_min if x_min < 0 else 0
    y_min = -y_min if y_min < 0 else 0
    translation_mat = np.array([[1, 0, x_min], [0, 1, y_min], [0, 0, 1]])
    H = np.dot(translation_mat, best_ransac_matrix_)
    warped_r = cv2.warpAffine(src=right, M=translation_mat[:2, :], dsize=size)
    warped_l = cv2.warpAffine(src=left, M=H[:2,:], dsize=size)
    stitched_image = np.maximum(warped_l, warped_r)
    return stitched_image

img1_path = "images/left.jpg"
img2_path = "images/right.jpg"

# Load images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Note: OpenCV uses BGR instead of RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Stitch images
stitchedImage1 = stitch_ext(img1, img2, 100)
stitchedImage2 = stitch_ext(img2, img1, 100)

plt.imshow(stitchedImage1)
plt.axis('off')
plt.show()
plt.close()

plt.imshow(stitchedImage2)
plt.axis('off')
plt.show()
plt.close()