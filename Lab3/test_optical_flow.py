import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy import signal
import os
import matplotlib
import scipy


def load_images(name_image_t0, name_image_t1, image_dir="./data/"):
    # Load the two images
    I_t0 = cv2.imread(image_dir + name_image_t0, 0)
    I_t1 = cv2.imread(image_dir + name_image_t1, 0)

    # Convert the to np.float32
    I_t0 = I_t0.astype(np.float32)
    I_t1 = I_t1.astype(np.float32)

    return I_t0, I_t1


def calculate_derivatives(I_t0, I_t1):
    """
    Obtain x, y and time derivatives of an image.
    """
    # --------------
    #  YOUR CODE HERE
    # --------------
    # Defining the filters
    I_t0 = I_t0 / 255
    I_t1 = I_t1 / 255
    # Convolve to get gradients w.r.to X, Y and T dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(I_t0, -1, kernel_x)  # Gradient over X
    fy = cv2.filter2D(I_t0, -1, kernel_y)  # Gradient over Y
    ft = cv2.filter2D(I_t1, -1, kernel_t) - cv2.filter2D(I_t0, -1, kernel_t)  # Gradient over Time

    return fx, fy, ft


# # Function that separates the image into subregions
# def calculate_subregions(I_t0, I_x, I_y, I_t, region_size):
#     """
#     input: I_t0, I_x, I_y, I_t, region_size
#     I_t0: image at time t0
#     I_x: image x-derivative
#     I_y: image y-derivative
#     I_t: image time derivative
#     region_size: size of the subregions

#     output: sub_I_x, sub_I_y, sub_I_t
#     sub_I_x: subregions of the image x-derivative
#     sub_I_y: subregions of the image y-derivative
#     sub_I_t: subregions of the image time derivative
#     """

#     # --------------
#     # YOUR CODE HERE
#     # --------------
#     height, width = I_t0.shape

#     sub_I_x = []
#     sub_I_y = []
#     sub_I_t = []

#     # Compute subregions based on region-size by looping over patches of the region size
#     for r in range(0, height, region_size):
#         for c in range(0, width, region_size):
#             sub_x, sub_y, sub_t = calculate_subregions_for_corners(I_x, I_y, I_t, r, c, region_size)
#
#             sub_I_x.append(sub_x)
#             sub_I_y.append(sub_y)
#             sub_I_t.append(sub_t)

#     return sub_I_x, sub_I_y, sub_I_t


# # Function that calulates subregions given corners
# def calculate_subregions_for_corners(I_x, I_y, I_t, r, c, region_size):
#     """
#     Input: I_x, I_y, I_t, r, c, region_size
#     I_x, I_y, I_t: image derivatives
#     r, c: corners of the subregions
#     region_size: size of the subregions

#     Output: sub_I_x, sub_I_y, sub_I_t
#     sub_I_x, sub_I_y, sub_I_t: subregions of the image derivatives
#     """
#     # --------------
#     # YOUR CODE HERE
#     # --------------
#     # Compute values for top right and bottem left corners and ensures that it does not exceed the image size
#     end_r = min(r + region_size, I_x.shape[0])
#     end_c = min(c + region_size, I_x.shape[1])

#     # Compute subregions
#     sub_I_x = I_x[r:end_r, c:end_c]
#     sub_I_y = I_y[r:end_r, c:end_c]
#     sub_I_t = I_t[r:end_r, c:end_c]

#     return sub_I_x, sub_I_y, sub_I_t


# Function that computes A, A.T and b for each subregion. Then, estimate
# optical flow (Vx, Vt) as given in Equation 22.
def calculate_flow_vectors(fx, fy, ft, feature_list, region_size, I_t0):
    """
    Calculate the local image flow vector (Vx, Vy) for each subregion by
    solving the linear system defined above.

    Input: I_x, I_y, I_t
    I_x, I_y, I_t: image derivatives

    Output: Vx, Vy
    Vx, Vy: Two lists containing, respectively, Vx, Vy of each subregion
    """
    # --------------
    #  YOUR CODE HERE
    # --------------
    u = np.zeros(I_t0.shape)
    v = np.zeros(I_t0.shape)
    w = int(region_size / 2)
    for feature in feature_list:  # for every corner
        j, i = feature.ravel()  # get cordinates of the corners (i,j). They are stored in the order j, i
        i, j = int(i), int(j)  # i,j are floats initially

        I_x = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
        I_y = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
        I_t = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

        b = np.reshape(I_t, (I_t.shape[0], 1))
        A = np.vstack((I_x, I_y)).T

        U = np.matmul(np.linalg.pinv(A), b)  # Solving for (u,v) i.e., U

        u[i, j] = U[0][0]
        v[i, j] = U[1][0]

    return (u, v)


# Function that plots the optical flow vectors
def plot_optical_flow(subregion_indices, V_x, V_y, I_t0):
    """
    Input: subregion_indices, V_x, V_y
    subregion_indices: indices of the subregions
    V_x, V_y: optical flow vectors
    """
    # --------------
    #  YOUR CODE HERE
    # --------------
    # x_coords = [center[0] for center in subregion_indices]
    # y_coords = [center[1] for center in subregion_indices]

    # x_coords = []
    # y_coords = []
    # for i in range(I_t0.shape[0]):
    #     for j in range(I_t0.shape[1]):
    #         x_coords.append((i, j))
    #         y_coords.append((int(i + V_x[i][j]), int(j + V_y[i][j])))

    X, Y = np.meshgrid(np.arange(0, I_t0.shape[0]), np.arange(0, I_t0.shape[1]))

    plt.figure(figsize=(10, 10))
    plt.imshow(I_t0, cmap='gray')  # Display the image
    plt.quiver(X, Y, V_x, V_y, angles='xy', scale_units='xy', scale=0.2, color='r')
    plt.title('Optical Flow')
    plt.show()

def optical_flow_demo(I_t0, I_t1, region_size=20):
    feature_list = cv2.goodFeaturesToTrack(I_t0, 10000, 0.05, 0.1)

    # Compute derivatives
    I_x, I_y, I_t = calculate_derivatives(I_t0, I_t1)

    # Calculate subregions
    #     sub_I_x, sub_I_y, sub_I_T = calculate_subregions(I_t0, I_x, I_y, I_t, region_size)

    # Calculate flow vectors
    (V_x, V_y) = calculate_flow_vectors(I_x, I_y, I_t, feature_list, region_size, I_t0)

    #     # Extract the V_x and V_y components from flow_vectors
    #     V_x = [vector[0] for vector in flow_vectors]
    #     V_y = [vector[1] for vector in flow_vectors]

    #     # Compute the centers of the subregions
    #     height, width = I_t0.shape

    #     # Determine centers based on region size
    #     x_centers = [(c + region_size/2) if c + region_size <= I_t0.shape[1] else (c + (I_t0.shape[1]-c)/2) for c in range(0, width, region_size)]
    #     y_centers = [(r + region_size/2) if r + region_size <= I_t0.shape[0] else (r + (I_t0.shape[0]-r)/2) for r in range(0, height, region_size)]

    #     # Create a grid of centers for every subregion
    #     X, Y = np.meshgrid(x_centers, y_centers)

    #     # Flatten the X, Y grid arrays for plotting
    #     x_coords = X.ravel()
    #     y_coords = Y.ravel()

    # Use the plot function to visualize optical flow
    # subregion_indices = []
    # for feature in feature_list:  # for every corner
    #     j, i = feature.ravel()  # get cordinates of the corners (i,j). They are stored in the order j, i
    #     i, j = int(i), int(j)
    #     subregion_indices.append((i, j))
    #
    # Compute the centers of the subregions
    height, width = I_t0.shape

    # Determine centers based on region size
    x_centers = [c for c in
                 range(0, width, region_size)]
    y_centers = [r for r in
                 range(0, height, region_size)]

    # Create a grid of centers for every subregion
    X, Y = np.meshgrid(x_centers, y_centers)

    # Flatten the X, Y grid arrays for plotting
    x_coords = X.ravel()
    y_coords = Y.ravel()

    # Use the plot function to visualize optical flow
    plot_optical_flow(list(zip(x_coords, y_coords)), V_x, V_y, I_t0)
    # plot_optical_flow(subregion_indices, V_x, V_y, I_t0)


# --------------
#  YOUR CODE HERE
# --------------
car_t0, car_t1 = load_images('car1.jpg', 'car2.jpg')
plant_t0, plant_t1 = load_images('plant1.png', 'plant2.png')
sphere_t0, sphere_t1 = load_images('sphere1.ppm', 'sphere2.ppm')

optical_flow_demo(car_t0, car_t1)
# optical_flow_demo(plant_t0, plant_t1)
# optical_flow_demo(sphere_t0, sphere_t1)