import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2


def gaussian(sigma, size):
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    g = np.exp(-x ** 2 / (2. * sigma ** 2))
    return g / g.sum()


def gaussian_derivative(gaussian_kernel):
    return np.gradient(gaussian_kernel)

def load_images(name_image_t0, name_image_t1, image_dir="./data/"):
    # Load the two images
    I_t0 = cv2.imread(image_dir + name_image_t0, 0)
    I_t1 = cv2.imread(image_dir + name_image_t1, 0)

    # Convert the to np.float32
    I_t0 = I_t0.astype(np.float32)
    I_t1 = I_t1.astype(np.float32)

    return I_t0, I_t1

def optical_flow(img1_path, img2_path):
    kernel_length = 9
    sigma = 0.5
    img1, img2 = load_images('plant1.png', 'plant2.png')
    # Gd = gaussian_derivative(gaussian(sigma, kernel_length))
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])

    Id = np.zeros((img1.shape[0], img1.shape[1], 3))
    Id[:, :, 0] = convolve(img1, kernel_x, mode='constant')
    Id[:, :, 1] = convolve(img1, kernel_y, mode='constant')
    Id[:, :, 2] = img2 - img1

    v = np.zeros((len(range(0, img1.shape[1], 15)), len(range(0, img1.shape[0], 15)), 2))
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)

    for x in range(0, img1.shape[1], 15):
        for y in range(0, img1.shape[0], 15):
            end_x = min(img1.shape[1] - x, 15)
            end_y = min(img1.shape[0] - y, 15)
            A = np.zeros((end_x * end_y, 2))
            b = np.zeros((end_x * end_y, 1))

            idx = 0
            for i in range(end_x):
                for j in range(end_y):
                    A[idx, :] = Id[j + y, i + x, 0:2]
                    b[idx, 0] = -Id[j + y, i + x, 2]
                    idx += 1

            # for i in range(w, I1g.shape[0] - w):
            #     for j in range(w, I1g.shape[1] - w):
            #         Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            #         Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            #         It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()


            LK = A
            LK_T = np.array(LK)  # transpose of A
            LK = np.array(np.matrix.transpose(LK))
            A1 = np.dot(LK_T, LK)  # Psedudo Inverse
            A2 = np.linalg.pinv(A1)
            A3 = np.dot(A2, LK_T)


            v_x = x // 15
            v_y = y // 15
            (u[v_x, v_y], v[v_x, v_y]) = np.dot(A3.T, b)
            # v[v_y, v_x, :], _ = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)

    # v[np.isnan(v)] = 0
    #
    # x, y = np.meshgrid(range(0, img1.shape[1], 15), range(0, img1.shape[0], 15))
    #
    # plt.figure()
    # plt.imshow(imread(img1_path), cmap='gray')
    # plt.quiver(x, y, u[::window_size, ::window_size], v[::window_size, ::window_size], color='r')
    # plt.show()
    window_size = 15
    plt.figure()
    plt.imshow(img1, cmap='gray')
    x, y = np.meshgrid(np.arange(0, img1.shape[1], window_size), np.arange(0, img1.shape[0], window_size))
    plt.quiver(x, y, u[::window_size, ::window_size], v[::window_size, ::window_size], color='r')
    plt.show()

# Example usage
optical_flow('./data/plant1.png', './data/plant2.png')
