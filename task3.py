from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import time
from task3_pca import do_pca
from task2 import kmeans_cluster


M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i
[H,W,L] = HICO_original.shape

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


X = image_cube_to_matrix(HICO_original)

print("X shape: ", X.shape)
P = 10
x_pca = do_pca(X, P)
print("X_pca shape: ", x_pca.shape)
HICO_pca = matrix_to_image_cube(x_pca, [H,W,P])
print("HICO_pca shape: ", HICO_pca.shape)

plt.figure()
plt.imshow(HICO_pca[:,:,0])
#plt.show()

start = time.time()
#kmeans_cluster(HICO_pca, "PCA_P_")
end = time.time()
print("Time elapsed: ", end - start)


#do_pca(np.random.rand(100, 1000), 10)