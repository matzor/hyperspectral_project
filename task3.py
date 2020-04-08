from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i

def image_cube_to_matrix(image_cube):
    [H,W,L] = image_cube.shape
    X = np.reshape(image_cube, [H*W,L])
    X = X.T
    return X

def matrix_to_image_cube(image_matrix, dimensions):
    return np.reshape(X.T, dimensions)

X = image_cube_to_matrix(HICO_original)
I = matrix_to_image_cube(X, HICO_original.shape)
