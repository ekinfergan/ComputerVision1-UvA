import math
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import os

import cv2
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.signal


def gauss1D(sigma, kernel_size):
    '''
    Generates the 1D Gaussian kernel with given sigma and kernel size.
    :param sigma: variance of the Gaussian function
    :param kernel_size: size of the kernel
    :return: 1D Gaussian kernel
    '''
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')

    # TODO: YOUR CODE HERE

    # assign zero to center of kernel
    center = kernel_size // 2

    for i in range(kernel_size):
        # find x coordinate for the given position in the kernel
        x = i - center
        # calculate the guassian value for this x
        x_gauss_val = (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(-1 * (x ** 2) / (2 * (sigma ** 2)))
        G[0, i] = x_gauss_val

    # normalize
    G = G / G.sum()

    return G

def gauss2D(sigma_x, sigma_y, kernel_size):
    '''
    Generates the 2D Gaussian kernel with given sigma_x for X dimension and sigma_y for Y dimension and kernel size.
    :param sigma_x: variance of the Gaussian function for dimension X
    :param sigma_y: variance of the Gaussian function for dimension Y
    :param kernel_size: size of the kernel
    :return: 2D Gaussian kernel
    '''
    # TODO: YOUR CODE HERE

    # checking whether the kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')

    # call gauss1D() for x-axis
    G_x = gauss1D(sigma_x, kernel_size)

    # call gauss1D() for y-axis
    G_y = gauss1D(sigma_y, kernel_size)

    # create a 2D array combining the above two:
    G = np.zeros((kernel_size, kernel_size))
    for x in range (kernel_size):
        for y in range(kernel_size):
            G[x, y] = G_x[0, x] * G_y[0, y]

    return G

def generateRotationMatrix(theta):
    '''
    Returns the rotation matrix for a given theta.
    Hint: https://en.wikipedia.org/wiki/Rotation_matrix
    :param theta: rotation parameter in radians
    :return: rotation matrix
    '''

    # TODO: YOUR CODE HERE
    # Code the rotation matrix which fits gabor equation given theta.
    rotMat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return rotMat

def createCos(rot_x, lamda, psi):
    '''
    Returns the 2D cosine carrier.
    '''

    # TODO: YOUR CODE HERE
    # Implement the cosine given rot_x, lamda and psi.
    cosCarrier = np.cos(2 * np.pi * rot_x / lamda + psi)

    return cosCarrier

def createSin(rot_x, lamda, psi):
    '''
    Returns the 2D sine carrier.
    '''

    # TODO: YOUR CODE HERE
    # Implement the sine given rot_x, lamda and psi.
    sinCarrier = np.sin(2 * np.pi * rot_x / lamda + psi)

    return sinCarrier


def createGauss(rot_x, rot_y, gamma, sigma):
    '''
    Returns the 2D Gaussian Envelope.
    Hint: ensure that gaussEnv has same dimensions as sinCarrier and cosCarrier by reshaping it in same manner
    '''

    # TODO: YOUR CODE HERE
    # Implement the Gaussian envelope.
    gaussEnv = np.exp(-(rot_x ** 2 + (gamma * rot_y) ** 2) / (2 * sigma ** 2))

    return gaussEnv

def createGabor(sigma, theta, lamda, psi, gamma):
    '''
    Creates a complex valued Gabor filter. Use it like this:
    myGabor = createGabor(sigma, theta, lamda, psi, gamma) generates Gabor kernels.

    :param sigma: Standard deviation of Gaussian envelope.
    :param theta: Orientation of the Gaussian envelope. Takes arguments in the range [0, pi/2).
    :param lamda: The wavelength for the carriers. The central frequency (w_c) of the carrier signals.
    :param psi: Phase offset for the carrier signal, sin(w_c . t + psi).
    :param gamma: Controls the aspect ratio of the Gaussian envelope
    :return: myGabor - A matrix of size [h,w,2], holding the real and imaginary
                        parts of the Gabor in myGabor(:,:,1) and myGabor(:,:,2), respectively.
    '''

    # Set the aspect ratio.
    sigma_x = sigma
    sigma_y = float(sigma)/gamma

    # Generate a grid
    nstds = 3
    xmax = max(abs(nstds*sigma_x*np.cos(theta)),abs(nstds*sigma_y*np.sin(theta)))
    xmax = np.ceil(max(1,xmax))
    ymax = max(abs(nstds*sigma_x*np.sin(theta)),abs(nstds*sigma_y*np.cos(theta)))
    ymax = np.ceil(max(1,ymax))

    # Make sure that we get square filters.
    xmax = max(xmax,ymax)
    ymax = max(xmax,ymax)
    xmin = -xmax
    ymin = -ymax

    # Generate a coordinate system in the range [xmin,xmax] and [ymin, ymax].
    [x,y] = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    # Convert to a 2-by-N matrix where N is the number of pixels in the kernel.
    XY = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)), axis=0)

    # Compute the rotation of pixels by theta.
    # Hint: Create appropriate rotation matrix to compute the rotated pixel coordinates: rot(theta) * XY.
    rotMat = generateRotationMatrix(theta)
    rot_XY = np.matmul(rotMat, XY)
    rot_x = rot_XY[0,:]
    rot_y = rot_XY[1,:]


    # Create the Gaussian envelope.
    # IMPLEMENT the helper function createGauss above.
    gaussianEnv = createGauss(rot_x, rot_y, gamma, sigma)

    # Create the orthogonal carrier signals.
    # IMPLEMENT the helper functions createCos and createSin above.
    cosCarrier = createCos(rot_x, lamda, psi)
    sinCarrier = createSin(rot_x, lamda, psi)

    # Modulate (multiply) Gaussian envelope with the carriers to compute
    # the real and imaginary components of the complex Gabor filter.
    myGabor_real = (gaussianEnv * cosCarrier).reshape(x.shape)  # TODO: modulate gaussianEnv with cosCarrier
    myGabor_imaginary = (gaussianEnv * sinCarrier).reshape(x.shape)   # TODO: modulate gaussianEnv with sinCarrier

    # Pack myGabor_real and myGabor_imaginary into myGabor.
    h, w = myGabor_real.shape
    myGabor = np.zeros((h, w, 2))
    myGabor[:,:,0] = myGabor_real
    myGabor[:,:,1] = myGabor_imaginary

    # Uncomment below to see how are the gabor filters
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(myGabor_real)    # Real
    # ax.axis("off")
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(myGabor_imaginary)    # Real
    # ax.axis("off")

    return myGabor

def load_image(image_id: str='Polar'):
    '''
    Loads an image, resizes image with proper resize factor and sets proper color representation
    :param image_id: id of an image: Kobi, Polar, Robin-1, Robin-2, Cows, SciencePark
    :return: image
    '''
    if image_id == 'Kobi':
        img = cv2.imread('./sample_data/kobi.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.25
    elif image_id == 'Polar':
        img = cv2.imread('./sample_data/polar-bear-hiding.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.75
    elif image_id == 'Robin-1':
        img = cv2.imread('./sample_data/robin-1.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 1
    elif image_id == 'Robin-2':
        img = cv2.imread('./sample_data/robin-2.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.5
    elif image_id == 'Cows':
        img = cv2.imread('./sample_data/cows.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.5
    elif image_id == 'SciencePark':
        img = cv2.imread('./sample_data/sciencepark.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.2
    else:
        raise ValueError('Image not available.')
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    return img


def show_image(image, image_id:str= "Polar", cmap='gray'):
    '''
    Displays image in grey scale
    :param image: image that should be displayed
    :param image_id: id of an image: Kobi, Polar, Robin-1, Robin-2, Cows, SciencePark
    :param cmap: matplotlib cmap arg
    '''
    # plt.figure()
    # plt.title(image_id)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

def filterBank(img):

  # Design array of Gabor Filters
  # In this code section, you will create a Gabor Filterbank. A filterbank is
  # a collection of filters with varying properties (e.g. {shape, texture}).
  # A Gabor filterbank consists of Gabor filters of distinct orientations
  # and scales. We will use this bank to extract texture information from the
  # input image.

  numRows, numCols = img.shape

  # Estimate the minimum and maximum of the wavelengths for the sinusoidal
  # carriers.
  # ** This step is pretty much standard, therefore, you don't have to
  #    worry about it. It is cycles in pixels. **
  lambdaMin = 4/np.sqrt(2)
  lambdaMax = np.sqrt(abs(numRows)**2 + abs(numCols)**2)

  # Specify the carrier wavelengths.
  # (or the central frequency of the carrier signal, which is 1/lambda)
  n = np.floor(np.log2(lambdaMax/lambdaMin))
  lambdas = 2**np.arange(0, (n-2)+1) * lambdaMin
  # lambdas = lambdas[:-2]
  lambdas = [lambdas[1], lambdas[2]]

  # Define the set of orientations for the Gaussian envelope.
  dTheta       = 2 * np.pi/8                  # \\ the step size
  orientations = np.arange(0, np.pi+dTheta, dTheta)
  # orientations = orientations[:-1]
  # orientations = orientations[:3]
  # Define the set of sigmas for the Gaussian envelope. Sigma here defines
  # the standard deviation, or the spread of the Gaussian.
  sigmas = np.array([5,6])

  # Now you can create the filterbank. We provide you with a Python list
  # called gaborFilterBank in which we will hold the filters and their
  # corresponding parameters such as sigma, lambda and etc.
  # ** All you need to do is to implement createGabor(). Rest will be handled
  #    by the provided code block. **
  gaborFilterBank = []
  tic = time.time()
  for lmbda in lambdas:
    for sigma in sigmas:
      for theta in orientations:
            # Filter parameter configuration for this filter.
            psi    = 0
            gamma  = 0.5

            # Create a Gabor filter with the specs above,
            # using the function createGabor()
            # and store result in gaborFilterBank

            filter_config = {}
            filter_config["filterPairs"] = createGabor( sigma, theta, lmbda, psi, gamma )
            filter_config["sigma"]       = sigma
            filter_config["lmbda"]       = lmbda
            filter_config["theta"]       = theta
            filter_config["psi"]         = psi
            filter_config["gamma"]       = gamma
            gaborFilterBank.append(filter_config)

  ctime = time.time() - tic

  print('--------------------------------------\n \t\tDetails\n--------------------------------------')
  print(f'Total number of filters       : {len(gaborFilterBank)}')
  print(f'Number of scales (sigma)      : {len(sigmas)}')
  print(f'Number of orientations (theta): {len(orientations)}')
  print(f'Number of carriers (lambda)   : {len(lambdas)}')
  print(f'---------------------------------------')
  print(f'Filter bank created in {ctime} seconds.')
  print(f'---------------------------------------')
  return(gaborFilterBank)


def gaborFeatures(img, gaborFilterBank, visFlag=False):
    '''
    Filter images using Gabor filter bank using quadrature pairs (real and imaginary parts)
    You will now filter the input image with each complex Gabor filter in
    gaborFilterBank structure and store the output in the cell called featureMaps.
    Hint-1: Apply both the real imaginary parts of each kernel
            separately in the spatial domain (i.e. over the image).
    Hint-2: Assign each output (i.e. real and imaginary parts) in
            variables called real_out and imag_out.
    Hint-3: Use built-in cv2 function, filter2D, to convolve the filter
            with the input image. Check the options for padding. Find
            the one that works well. You might want to
            explain what works better and why shortly in the report.
    '''

    featureMaps = []

    for gaborFilter in gaborFilterBank:
        # gaborFilter["filterPairs"] has two elements. One is related to the real part
        # of the Gabor Filter and the other one is the imagineray part.

        real_out = cv2.filter2D(img, cv2.cv2.CV_32F, gaborFilter["filterPairs"][:, :, 0]) # cv2.CV_8U
        # real_out = scipy.signal.convolve2d(img, gaborFilter["filterPairs"][:, :, 0], boundary='wrap', mode='same') # cv2.CV_8U
        imag_out = cv2.filter2D(img, cv2.cv2.CV_32F, gaborFilter["filterPairs"][:, :, 1])
        # imag_out = scipy.signal.convolve2d(img, gaborFilter["filterPairs"][:, :, 1], boundary='wrap', mode='same')
        #     real_out = cv2.filter2D(img, -1, gaborFilter["filterPairs"][:, :, 0])   # \\TODO: filter the grayscale input with real part of the Gabor
        #     imag_out = cv2.filter2D(img, -1, gaborFilter["filterPairs"][:, :, 1])  # \\TODO: filter the grayscale input with imaginary part of the Gabor

        featureMaps.append(np.stack((real_out, imag_out), 2))

        # Visualize the filter responses if you wish.
        if visFlag:
            fig = plt.figure()

            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(real_out)  # Real
            title = "Re[h(x,y)], \n lambda = {0:.4f}, \n theta = {1:.4f}, \n sigma = {2:.4f}".format(
                gaborFilter["lmbda"],
                gaborFilter["theta"],
                gaborFilter["sigma"])
            ax.set_title(title)
            ax.axis("off")

            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(imag_out)  # Real
            title = "Im[h(x,y)], \n lambda = {0:.4f}, \n theta = {1:.4f}, \n sigma = {2:.4f}".format(
                gaborFilter["lmbda"],
                gaborFilter["theta"],
                gaborFilter["sigma"])
            ax.set_title(title)
            ax.axis("off")
            plt.show()

    # Compute the magnitude
    # Now, you will compute the magnitude of the output responses.
    # \\ Hint: (real_part^2 + imaginary_part^2)^(1/2) \\
    featureMags = []
    for i, fm in enumerate(featureMaps):
        real_part = fm[..., 0]
        imag_part = fm[..., 1]
        mag = np.sqrt(real_part ** 2 + imag_part ** 2)  # \\TODO: Compute the magnitude here
        featureMags.append(mag)

        # Visualize the magnitude response if you wish.
        if visFlag:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mag.astype(np.uint8))  # visualize magnitude
            title = "Re[h(x,y)], \n lambda = {0:.4f}, \n theta = {1:.4f}, \n sigma = {2:.4f}".format(
                gaborFilterBank[i]["lmbda"],
                gaborFilterBank[i]["theta"],
                gaborFilterBank[i]["sigma"])
            ax.set_title(title)
            ax.axis("off")

    print('Created', len(featureMags), 'features for each pixel')
    return featureMags

def clusterFeatures(img, featureMags, smoothingFlag=True):

  '''
  Prepare and Preprocess features
  You can think of each filter response as a sort of feature representation
  for the pixels. Now that you have numFilters = |gaborFilterBank| filters,
  we can represent each pixel by this many features.
  Question: What kind of features do you think gabor filters might correspond to?

  You will now implement a smoothing operation over the magnitude images in
  featureMags.
  Hint-1: For each i in [1, length(featureMags)], smooth featureMags{i} using an appropriate first order Gaussian kernel.
  Hint-2: cv2 filter2D function is helpful here.
  '''

  numRows, numCols = img.shape
  features = np.zeros(shape=(numRows, numCols, len(featureMags)))

  if smoothingFlag:
    # \\TODO:
    # FOR_LOOP
        # i)  filter the magnitude response with appropriate Gaussian kernels
        # ii) insert the smoothed image into features[:,:,j]
    # END_FOR
    lambdaMin = 4 / np.sqrt(2)
    lambdaMax = np.sqrt(abs(numRows) ** 2 + abs(numCols) ** 2)
    n = np.floor(np.log2(lambdaMax / lambdaMin))
    lambdas = 2 ** np.arange(0, (n - 2) + 1) * lambdaMin
    lambdas = lambdas[:-2]
    lambdas = np.tile(lambdas, 8)

    sigma = 0.2
    kernel_size = 3
    for i, fm in enumerate(featureMags):
        sigma_ = sigma#kernel_size*sigma*lambdas[i]
        # g_kernel = gauss2D(sigma, sigma, kernel_size)
        # features[:,:,i] = cv2.filter2D(fm, -1, g_kernel)
        # features[:,:,i] = scipy.signal.convolve2d(fm, g_kernel, boundary='symm', mode='same')
        features[:,:,i] = scipy.ndimage.gaussian_filter(fm, sigma=sigma_)
  else:
    # Don't smooth but just insert magnitude images into the matrix called features.
    for i, fmag in enumerate(featureMags):
        features[:,:,i] = fmag
        if i//5==0:
          print(i)

  '''
  Reshape the filter outputs (i.e. tensor called features) of size
  [numRows, numCols, numFilters] into a matrix of size [numRows*numCols, numFilters]
  This will constitute our data matrix which represents each pixel in the
  input image with numFilters features.
  '''

  # Adding the spatial information in the features
  # x_dim = np.arange(1, numCols + 1)
  # y_dim = np.arange(1, numRows + 1)
  # X_dim, Y_dim = np.meshgrid(x_dim, y_dim)
  # features = np.dstack((features, X_dim, Y_dim))

  features = np.reshape(features, newshape=(numRows * numCols, -1))

  '''
  Standardize features.
  Hint: see https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
  '''
  # features = None  # \\ TODO: i)  Implement standardization on matrix called features.
                     #          ii) Return the standardized data matrix.
#   print(np.mean(features, axis=0).shape)
  features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)


  '''
  (Optional) Visualize the saliency map using the first principal component
  of the features matrix. It will be useful to diagnose possible problems
  with the pipeline and filterbank.
  '''

  transformed_feature = PCA(n_components=1).fit_transform(features) # select the first component
  transformed_feature = np.ascontiguousarray(transformed_feature, dtype=np.float32)
  feature2DImage = np.reshape(transformed_feature,newshape=(numRows,numCols))
  # plt.figure()
  # plt.title(f'Pixel representation projected onto first PC')
  # plt.imshow(feature2DImage, cmap='gray')
  # plt.axis("off")
  # plt.show()
  '''
  Apply k-means algorithm to cluster pixels using the data matrix features.
  Hint-1: search about sklearn kmeans function https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.
  Hint-2: when calling sklearn's kmeans function, use the parameter n_clusters as defined in the aloritm description above.
  '''
  n_clusters = 2  # Number of clusters (K)
  kmeans = KMeans(n_clusters=n_clusters)
  tic = time.time()
  pixLabels = kmeans.fit_predict(transformed_feature)  # \\TODO: Return cluster labels per pixel
  ctime = time.time() - tic
  print(f'Clustering completed in {ctime} seconds.')
  return pixLabels


image_id = "Kobi"
img = load_image(image_id)  # load an image with the Polar bear
# show_image(img, f'Input image: {image_id}')
if len(img.shape) != 2:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
# show_image(img, f'Input image: {image_id}')
gaborFilterBank = filterBank(img)
featureMags = gaborFeatures(img, gaborFilterBank, False)
pixLabels = clusterFeatures(img, featureMags, True)

numRows, numCols = img.shape
pixLabels = np.reshape(pixLabels, newshape=(numRows, numCols))
# plt.imshow(pixLabels)
# plt.axis("off")
# plt.show()
# Use the pixLabels to visualize segmentation.
Aseg1 = np.zeros_like(img)
Aseg2 = np.zeros_like(img)
BW = pixLabels == 1  # check for the value of your labels in pixLabels (could be 1 or 0 instead of 2)
# BW = np.repeat(BW[:, :, np.newaxis], 3, axis=2) # do this only if you have 3 channels in the img
Aseg1[BW] = img[BW]
Aseg2[~BW] = img[~BW]
plt.imshow(Aseg1, 'gray', interpolation='none')
plt.imshow(Aseg2, 'jet',  interpolation='none', alpha=0.7)
plt.axis("off")
plt.show()