import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import ransac
print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()


import cv2
import numpy as np

def stitchImages(left, right, N=800):
    print("stitching image ...")
    keypoints_1, keypoints_2, matches = ransac.keypoint_matching(left, right)
    best_ransac_matrix, _ = ransac.ransac(keypoints_1, keypoints_2, matches, N)
    best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)

    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
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
    warped_l = cv2.warpAffine(src=left, M=H[:2, :], dsize=size)
    height_r, width_r, channel_r = right.shape
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    warped_r = cv2.warpAffine(src=right, M=translation_mat[:2, :], dsize=size)

    # Pad the images to ensure they have the same size for stitching
    pad_x = max(warped_l.shape[1], warped_r.shape[1]) - warped_l.shape[1]
    pad_y = max(warped_l.shape[0], warped_r.shape[0]) - warped_l.shape[0]
    warped_l = cv2.copyMakeBorder(warped_l, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    warped_r = cv2.copyMakeBorder(warped_r, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])


    black = np.zeros(3)  # Black pixel.
    stitch_image = np.zeros_like(warped_l)

    # Stitching procedure, store results in stitch_image.
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                stitch_image[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                stitch_image[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                stitch_image[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                stitch_image[i, j, :] = black

    return stitch_image


def stitchImages2(left, right, N=800):
    """
    Given two input images, return the stitched image.
    Arguments:
      img1: the first image (in RGB)
      img2: the second image (in RGB)
      N: number of iterations for RANSAC
    Returns:
      The stitched image of the two images
    """
    # TODO: 1. Find the best transformation.
    keypoints_1, keypoints_2, matches = ransac.keypoint_matching(left, right)
    best_ransac_matrix, _ = ransac.ransac(keypoints_1, keypoints_2, matches, N)
    best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)

    # TODO: 2. Estimate the size of the stitched image.
    # Hint: Calculate the transformed coordinates of corners of the *right.jpg*
    print("stiching image ...")
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # right image
    height_r, width_r, channel_r = right.shape
    corners = [[0, 0, 1], [width_r, 0, 1], [width_r, height_r, 1], [0, height_r, 1]]
    corners_new = [np.dot(best_ransac_matrix_, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)
    # translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # H = np.dot(translation_mat, best_ransac_matrix_) #????
    # Get height, width
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    # stitch_image = cv2.warpAffine(right, best_ransac_matrix, (width_new, height_new))

    stitch_image = cv2.warpAffine(right, best_ransac_matrix, (right.shape[1] +
                                                   int(best_ransac_matrix[0, 2] + 1), right.shape[0]))
    stitch_image[:, :left.shape[1]] = left

    # right image
    # warped_r = cv2.warpPerspective(src=right, M=H, dsize=size)




    ################################################
    # left image
    # height_l, width_l, channel_l = left.shape
    # corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    # best_ransac_matrix_ = np.concatenate((best_ransac_matrix, np.array([0, 0, 1]).reshape(1, -1)), axis=0)
    # corners_new = [np.dot(best_ransac_matrix_, corner) for corner in corners]
    # corners_new = np.array(corners_new).T
    # x_news = corners_new[0] / corners_new[2]
    # y_news = corners_new[1] / corners_new[2]
    # y_min = min(y_news)
    # x_min = min(x_news)

    # translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # H = np.dot(translation_mat, best_ransac_matrix_)

    # # Get height, width
    # height_new = int(round(abs(y_min) + height_l))
    # width_new = int(round(abs(x_min) + width_l))
    # size = (width_new, height_new)

    # right image
    # warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
#### lolo
    # height_r, width_r, channel_r = right.shape
    #
    # height_new = int(round(abs(y_min) + height_r))
    # width_new = int(round(abs(x_min) + width_r))
    # size = (width_new, height_new)
    #
    # warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
    #
    # h1, w1 = left.shape[:2]
    # h2, w2 = right.shape[:2]
    # corners_img2 = np.array([[0, 0], [0, h2], [w2, 0], [w2, h2]])
    #
    # black = np.zeros(3)  # Black pixel.
    #
    # # Stitching procedure, store results in warped_l.
    # for i in range(warped_r.shape[0]):
    #     for j in range(warped_r.shape[1]):
    #         pixel_l = warped_l[i, j, :]
    #         pixel_r = warped_r[i, j, :]
    #
    #         if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
    #             warped_l[i, j, :] = pixel_l
    #         elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
    #             warped_l[i, j, :] = pixel_r
    #         elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
    #             warped_l[i, j, :] = (pixel_l + pixel_r) / 2
    #         else:
    #             pass
    #
    # stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]

    return stitch_image

img1_path = "images/left.jpg"
img2_path = "images/right.jpg"

# Load images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Note: OpenCV uses BGR instead of RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Stitch images
stitchedImage1 = stitchImages(img1, img2, 100)
# stitchedImage1 = stitchImages(img2, img1, 100)

plt.imshow(stitchedImage1)
plt.axis('off')
plt.show()
plt.close()