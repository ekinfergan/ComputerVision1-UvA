import sys
import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

def load_images(name_image_t0, name_image_t1, image_dir="./data/"):
    # Load the two images
    I_t0 = cv2.imread(image_dir + name_image_t0, 0)
    I_t1 = cv2.imread(image_dir + name_image_t1, 0)

    # Convert the images to np.float32
    I_t0 = I_t0.astype(np.float32)
    I_t1 = I_t1.astype(np.float32)

    return I_t0, I_t1


def calculate_derivatives(I_t0, I_t1):
    """
    Obtain x, y and time derivatives of an image.
    """
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    I_t0 = I_t0 / 255.  # normalize pixels
    I_t1 = I_t1 / 255.  # normalize pixels
    fx = signal.convolve2d(I_t0, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(I_t0, kernel_y, boundary='symm', mode='same')
    ft = signal.convolve2d(I_t1, kernel_t, boundary='symm', mode='same') + \
         signal.convolve2d(I_t0, -kernel_t, boundary='symm', mode='same')
    return fx, fy, ft


# Function that separates the image into subregions
def calculate_subregions(I_t0, I_x, I_y, I_t, region_size):
    """
    input: I_t0, I_x, I_y, I_t, region_size
    I_t0: image at time t0
    I_x: image x-derivative
    I_y: image y-derivative
    I_t: image time derivative
    region_size: size of the subregions

    output: sub_I_x, sub_I_y, sub_I_t
    sub_I_x: subregions of the image x-derivative
    sub_I_y: subregions of the image y-derivative
    sub_I_t: subregions of the image time derivative
    """
    sub_I_x = []
    sub_I_y = []
    sub_I_t = []
    w = region_size // 2
    for i in range(w, I_t0.shape[0] - w):
        for j in range(w, I_t0.shape[1] - w):
            sub_I_x.append(I_x[i - w:i + w + 1, j - w:j + w + 1].flatten())
            sub_I_y.append(I_y[i - w:i + w + 1, j - w:j + w + 1].flatten())
            sub_I_t.append(I_t[i - w:i + w + 1, j - w:j + w + 1].flatten())

    return sub_I_x, sub_I_y, sub_I_t


# Function that calulates subregions given corners
def calculate_subregions_for_corners(I_x, I_y, I_t, r, c, region_size):
    """
    Input: I_x, I_y, I_t, r, c, region_size
    I_x, I_y, I_t: image derivatives
    r, c: corners of the subregions
    region_size: size of the subregions

    Output: sub_I_x, sub_I_y, sub_I_t
    sub_I_x, sub_I_y, sub_I_t: subregions of the image derivatives
    """

    # Compute values for top right and bottem left corners and ensures that it does not exceed the image size
    end_r = min(r + region_size, I_x.shape[0])
    end_c = min(c + region_size, I_x.shape[1])

    # Compute subregions
    sub_I_x = I_x[r:end_r, c:end_c]
    sub_I_y = I_y[r:end_r, c:end_c]
    sub_I_t = I_t[r:end_r, c:end_c]

    return sub_I_x, sub_I_y, sub_I_t


# Function that computes A, A.T and b for each subregion. Then, estimate
# optical flow (Vx, Vt) as given in Equation 22.
def calculate_flow_vectors(I_x, I_y, I_t):
    """
    Calculate the local image flow vector (Vx, Vy) for each subregion by
    solving the linear system defined above.
    """
    flow_vectors = []
    for sub_x, sub_y, sub_t in zip(I_x, I_y, I_t):
        A = np.vstack((sub_x, sub_y)).T
        b = sub_t
        A_p = np.linalg.pinv(A)
        flow_vectors.append(np.dot(A_p, b))
    return flow_vectors


def plot_optical_flow(subregion_centers, V_x, V_y, I_t0):
    """
    Input: subregion_centers, V_x, V_y
    subregion_centers: list of tuples with x, y coordinates of subregion centers
    V_x, V_y: optical flow vectors
    """
    #     for i in range(w, I_t0.shape[0] - w):
    #         for j in range(w, I_t0.shape[1] - w)
    #
    # x_coords = [center[0] for center in subregion_centers]
    # y_coords = [center[1] for center in subregion_centers]

    plt.figure(figsize=(10, 10))
    plt.imshow(I_t0, cmap='gray')  # Display the image
    plt.quiver(subregion_centers[0], subregion_centers[1], V_x, V_y, color='r', angles='xy', scale_units='xy', scale=0.4)
    plt.title('Optical Flow')
    plt.show()


#     plt.figure()
#     plt.imshow(car_t0, cmap='gray')
#     x, y = np.meshgrid(np.arange(0, car_t0.shape[1], window_size), np.arange(0, car_t0.shape[0], window_size))
#     plt.quiver(x, y, u[::window_size, ::window_size], v[::window_size, ::window_size], color='r', angles='xy', scale_units='xy', scale=0.4)
#     plt.show()

def optical_flow_demo(I_t0, I_t1, region_size=20):
    # Compute derivatives
    Gx, Gy, Gt = calculate_derivatives(I_t0, I_t1)

    # Calculate subregions
    sub_I_x, sub_I_y, sub_I_T = calculate_subregions(I_t0, Gx, Gy, Gt, region_size)

    # Calculate flow vectors
    flow_vectors = calculate_flow_vectors(sub_I_x, sub_I_y, sub_I_T)

    w = region_size // 2
    u = np.zeros(I_t0.shape)
    v = np.zeros(I_t0.shape)
    k = 0
    for i in range(w, I_t0.shape[0] - w):
        for j in range(w, I_t0.shape[1] - w):
            (u[i, j], v[i, j]) = flow_vectors[k]
            k += 1
    x, y = np.meshgrid(np.arange(0, I_t0.shape[1], region_size), np.arange(0, I_t0.shape[0], region_size))
    plot_optical_flow((x, y), u[::region_size, ::region_size], v[::region_size, ::region_size], I_t0)
    # plt.figure()
    # plt.imshow(I_t0, cmap='gray')
    #
    # plt.quiver(x, y, u[::region_size, ::region_size], v[::region_size, ::region_size], color='r', angles='xy',
    #            scale_units='xy', scale=0.4)
    # plt.show()

    # Extract the V_x and V_y components from flow_vectors
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

    # x, y = np.meshgrid(np.arange(0, car_t0.shape[1], window_size), np.arange(0, car_t0.shape[0], window_size))
    #
    # # Use the plot function to visualize optical flow
    # plot_optical_flow(list(zip(x_coords, y_coords)), V_x, V_y, I_t0)


car_t0, car_t1 = load_images('car1.jpg', 'car2.jpg')
# plant_t0, plant_t1 = load_images('plant1.png', 'plant2.png')
# sphere_t0, sphere_t1 = load_images('sphere1.ppm', 'sphere2.ppm')

optical_flow_demo(car_t0, car_t1)
# optical_flow_demo(plant_t0, plant_t1)
# optical_flow_demo(sphere_t0, sphere_t1)
