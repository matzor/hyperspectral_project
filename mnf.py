import numpy as np
from scipy.linalg import eig
import spectral as spy


def image_cube_to_matrix(image_cube):
    """ 
    Input: 
        image_cube: Hyperspectral image data, np.array shape W x H x L
    Returns:
        X: input reshaped to matrix X with L rows and N = WH columns
    """
    [H,W,L] = image_cube.shape
    X = np.reshape(image_cube, [H*W,L])
    X = X.T
    return X

def matrix_to_image_cube(image_matrix, dimensions):
    """
    Input: 
        image_matrix: matrix X with L rows and N = WH columns
        dimensions: [height, width, length], ie. HICO.shape()
    Returns: 
        image_cube: input reshaped to dimensions
    """
    return np.reshape(image_matrix.T, dimensions)

def error(X, X_ref):
    return np.mean(np.linalg.norm(X - X_ref, axis=0)) / \
        np.mean(np.linalg.norm(X_ref, axis=0))

def mnf(X, P, sigma_n, sigma):
    """ 
    Performs Maximum Noise Fraction on X
    
    Inputs:
        X: Noisy signal, np.array L x N
        P: Number of components
        sigma_n: covariance of noise
        sigma: covariance of signal

    Returns:
        x_hat: Maximum Noise Fraction 
    """

    sigmas = sigma_n @ np.linalg.inv(sigma)
    w, vl, vr = eig(sigmas, left=True, right=True)
    w = np.sort(w)
    #vl = np.sort(vl)
    #vr = np.sort(vr)

    Y = vl.T @ X

    x_hat = vr[:,:P] @ Y[:P, :]
    return x_hat

def estimate_noise(X):
    X_n = np.empty((X.shape[0], X.shape[1]-1))
    for i in range(X_n.shape[1]):
        X_n[:, i] = X[:,i] - X[:,i+1]
    sigma_n = np.cov(X_n)
    return sigma_n
    

