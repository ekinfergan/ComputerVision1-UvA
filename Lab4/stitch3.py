import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import ransac
print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()



def calc_padding(img_1, img_2):
    pad_1 = (max(img_1.shape[0] - img_2.shape[0], 0), max(img_1.shape[1] - img_2.shape[1], 0))
    pad_2 = (max(img_2.shape[0] - img_1.shape[0], 0), max(img_2.shape[1] - img_1.shape[1], 0))
    padded_1 = np.pad(img_1, ((0, pad_1[0]), (0, pad_1[1]), (0, 0)))
    padded_2 = np.pad(img_2, ((0, pad_2[0]), (0, pad_2[1]), (0, 0)))
    return padded_1, padded_2

# pad_1 = [max(size(img_1,1)-size(img_2,1),0) max(size(img_1,2)-size(img_2,2),0)];
# pad_2 = [max(size(img_2,1)-size(img_1,1),0) max(size(img_2,2)-size(img_1,1),0)];
# padded_1 = padarray(img_1, pad_2,'post');
# padded_2 = padarray(img_2, pad_1,'post');

def stitch(right, left, N=800):
    keypoints_1, keypoints_2, matches = ransac.keypoint_matching(right, left)
    best_ransac_matrix, _ = ransac.ransac(keypoints_1, keypoints_2, matches, N)
    # best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)

    left_img_padded, right_img_padded = calc_padding(left, right)

    img_right_trans = ransac.affine_transform(right, best_ransac_matrix, warp='inverse')
    [trans_img_padded, left_img_padded] = calc_padding(img_right_trans, left)
    # stitched = max(trans_img_padded, left_img_padded)
    stitched_image = np.maximum(trans_img_padded, left)


    # original = np.concatenate((left_img_padded, right_img_padded), axis=1)
    return stitched_image

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

def stitch_ext2(right, left, N=800):
    keypoints_1, keypoints_2, matches = ransac.keypoint_matching(right, left)
    best_ransac_matrix, _ = ransac.ransac(keypoints_1, keypoints_2, matches, N)
    best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)

    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)
    corners_new = [np.dot(best_ransac_matrix_, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, best_ransac_matrix_)

    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape

    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
    stitched_image = np.maximum(warped_r, warped_l)
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

# warped_l = cv2.warpAffine(src=left, M=H[:2, :], dsize=size)
# warped_r = cv2.warpAffine(src=right, M=H[:2, :], dsize=size)
# warped_l = cv2.warpAffine(src=left, M=H[:2, :], dsize=size)
# warped_r = cv2.warpAffine(src=right, M=translation_mat[:2, :], dsize=size)


# left_img_padded, right_img_padded = calc_padding( warped_l, warped_r)  # Assuming calc_padding function works correctly
# img_right_trans = ransac.affine_transform(right_img_padded, best_ransac_matrix, warp='inverse')  # Assuming this function works correctly
# trans_img_padded, left_img_padded = calc_padding(warped_r, left_img_padded)  # Assuming calc_padding function works correctly
# warped_l = np.ones(warped_l.shape)