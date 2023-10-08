import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()

def keypoint_matching(image1, image2):
  """
    Given two input images, find and return the matching keypoints.
    Arguments:
    image1: the first image (in RGB)
    image2: the second image (in RGB)
    Returns:
    The keypoints of image1, the keypoints of image2 and the matching
    keypoints between the two images
  """

  print('\nFinding matching features...')
  # ================
  img1_gray   = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  img2_gray   = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

  # Getting the Interest Pointss
  keypoints_1, desc1 = sift.detectAndCompute(img1_gray, None)
  keypoints_2, desc2 = sift.detectAndCompute(img2_gray, None)

  bf      = cv2.BFMatcher(crossCheck = True)
  matches = bf.match(desc1, desc2)
  # ================

  print("Number of keypoints in image 1:        ", len(keypoints_1))
  print("Number of keypoints in image 2:        ", len(keypoints_2))
  print("Number of keypoints after matching:    ", len(matches), "\n")

  return keypoints_1, keypoints_2, matches


def visualization(src_img, ref_img, best_matrix):
    """
    Given the source image and the reference image, visualize:
    1. The transformed src image using the forward warping
    2. The transformed src image using the inverse warping
    3. The transformed src image using OpenCV's warpAffine function
    4. The original reference image
    Arguments:
      - src_img: the image to transform
      - ref_img: the referenced image to transform to
      - best_matrix: the best transformation matrix
    """
    # NOTE: if you input a colored image, it expects RGB and not BGR colors.

    # ================
    # Custom Method
    trs_f = affine_transform(src_img, best_matrix,warp='inverse')
    trs_b = affine_transform(src_img, best_matrix, warp='inverse')

    # OpenCVs Method
    r, c, _ = src_img.shape
    trs_cv = cv2.warpAffine(src_img, best_matrix[0:2], (c, r))

    # Plotting
    images = [trs_f, trs_b, trs_cv, ref_img]
    titles = ["Forward Warping", "Inverse Warping", "OpenCV's warpAffine", "Original Reference"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 8))

    for idx, image in enumerate(images):
        ax[idx].imshow(image)
        ax[idx].axis('off')
        ax[idx].set_title(titles[idx])

    # Styling
    NUM = 1
    IMG1  =1
    IMG2 = 2
    fig.suptitle(f"Figure {NUM}) Visualization of the Image Transformation Results from {IMG1} to {IMG2}", fontsize=14,
                 weight='bold')
    text = "Keypoint matching between all the points of interest between both Science Park Images."
    plt.figtext(0.51, 0.13, text, wrap=True, horizontalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()
    # ================

# Complete the function below. Use it in the `ransac` function and/or 'visualization' function.
def affine_transform(img, mat, warp='forward'):
    """
    Arguments:
      img: the first/second image (img1 or img2)
      mat: transformation matrix
      warp: forward or inverse warping
    Returns:
      transformed image
    """

    # =========================
    # OUR CODE HERE IF NEEDED
    # =========================
    h, w, _ = img.shape
    img_transformed = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
    if warp == 'forward':
        for y in range(h):
            for x in range(w):
                input_coords = np.array([x, y, 1])
                new_xy = np.floor(mat @ input_coords).astype(int)
                new_x, new_y = new_xy[0], new_xy[1]
                # Check if the new coordinates are within the image boundaries
                if 0 <= new_x < w and 0 <= new_y < h:
                    img_transformed[new_y, new_x, :] = img[y, x, :]
    if warp == 'inverse':
        mat_ = np.concatenate((mat, np.array([0, 0, 1]).reshape(1, -1)), axis=0)
        mat_inv = np.linalg.inv(mat_)[0:2]
        for y in range(h):
            for x in range(w):
                output_coords = np.array([x, y, 1])
                new_xy = np.floor(mat_inv @ output_coords).astype(int)
                new_x, new_y = new_xy[0], new_xy[1]

                # Check if the input coordinates are within the image boundaries
                if 0 <= new_x < w and 0 <= new_y < h:
                    img_transformed[y, x, :] = img[new_y, new_x, :]
    return img_transformed


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
    best_matrix = np.zeros((3, 3))
    P = 10
    random.seed(10)
    num_matches = len(matches)
    for i in range(N):
        print(f"\nItr No:{i + 1}")
        # index = np.random.randint(0, num_matches, P)
        index = [i for i in random.sample(range(0, num_matches), P)]
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
        if num_inliers > num_best_inliers:
            num_best_inliers = num_inliers
            best_matrix = affine_params.copy()

    print("Total number of matches: ", len(matches))
    print("Inliers found:           ", num_best_inliers)
    print("Outliers removed:        ", len(matches) - num_best_inliers)

    lol = 0
    return best_matrix, lol

img1_path = "images/sp1.jpg"
img2_path = "images/sp2.jpg"
#
# # Open images
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)
#
# # Note: OpenCV uses BGR instead of RGB
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#
# keypoints_1, keypoints_2, matches = keypoint_matching(img1,img2)
# best_matrix, _ = ransac(keypoints_1, keypoints_2, matches, 10)
# visualization(img1, img2, best_matrix)