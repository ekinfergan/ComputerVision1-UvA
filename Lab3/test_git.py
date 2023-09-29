import numpy as np
from scipy import signal
import cv2

def optical_flow(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  #*.25
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode='same')
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode='same') + \
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode='same')
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)

    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            A = np.vstack((Ix, Iy)).T
            b = It
            A2 = np.linalg.pinv(A)
            (u[i, j], v[i, j]) = np.dot(A2, b)  # we have the vectors with minimized square error

    return (u, v)

# Example usage
# Generate I1g and I2g (grayscale images)
def load_images(name_image_t0, name_image_t1, image_dir="./data/"):
    # Load the two images
    I_t0 = cv2.imread(image_dir + name_image_t0, 0)
    I_t1 = cv2.imread(image_dir + name_image_t1, 0)

    # Convert the to np.float32
    I_t0 = I_t0.astype(np.float32)
    I_t1 = I_t1.astype(np.float32)

    return I_t0, I_t1

# car_t0, car_t1 = load_images('sphere1.ppm', 'sphere2.ppm')
# car_t0, car_t1 = load_images('plant1.png', 'plant2.png')
car_t0, car_t1 = load_images('car2.jpg', 'car1.jpg')
window_size = 20
u, v = optical_flow(car_t0, car_t1, window_size=window_size, tau=1e-2)

# Plot optical flow
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(car_t0, cmap='gray')
x, y = np.meshgrid(np.arange(0, car_t0.shape[1], window_size), np.arange(0, car_t0.shape[0], window_size))
plt.quiver(x, y, u[::window_size, ::window_size], v[::window_size, ::window_size], color='r', angles='xy', scale_units='xy', scale=0.4)
plt.show()
