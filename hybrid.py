import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    result = np.empty(img.shape)
    
    if len(img.shape) == 2:
        h, w = img.shape
        c = 1
        img = np.expand_dims(img, axis=2)
    else:
        h, w, c = img.shape

    m, n = kernel.shape
    img_pad = np.zeros(( h + m - 1, w + n - 1, c))

    
    n1 = (n - 1) / 2
    m1 = (m - 1) / 2
    img_pad[m1:h+m1, n1:w+n1] = img

    
    kernel = kernel.reshape(-1)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.dot(kernel, np.reshape(img_pad[i:i+2*m1+1, j:j+2*n1+1], (m*n, c)))

    return result

   


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, np.fliplr(np.flipud(kernel)))

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    x_margin, y_margin = (width - 1) / 2, (height - 1) / 2

    x = np.arange(-x_margin, x_margin + 1, 1.0) ** 2
    y = np.arange(-y_margin, y_margin + 1, 1.0) ** 2

    coeff = 1 / (2 * sigma * sigma * np.pi)

    x_gaus = np.exp(-x / (2 * sigma * sigma))
    y_gaus = np.exp(-y / (2 * sigma * sigma)) * coeff

    output = np.outer(x_gaus, y_gaus)
   
    kernel = output / (np.sum(x_gaus) * np.sum(y_gaus))

    return kernel

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


