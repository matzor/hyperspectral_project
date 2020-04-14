from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import time
from task3_pca import do_pca
from task2 import kmeans_cluster
from mnf import mnf, error, estimate_noise, matrix_to_image_cube, image_cube_to_matrix

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i
[H,W,L] = HICO_original.shape


def task3c():
    """ 
    Runs PCA on HICO image cube, then kmeans on compressed data """
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


#task2c()

def task3e():
    P = [100, 50, 30, 10, 5, 3, 1]
    error_mnf = np.zeros(len(P))
    error_pca = np.zeros(len(P))
    X_n = image_cube_to_matrix(HICO_noisy)
    X = image_cube_to_matrix(HICO_original)
    sigma_n = estimate_noise(X_n)
    sigma = np.cov(X_n)

    # Comparing original noisy to original:
    error_noise = error(X_n, X)
    print("Noisy error, no filtering: ", error_noise)

    # Comparing error between varying amount of components
    for i in range(len(P)):
        X_hat_mnf = mnf(X_n, P[i], sigma_n, sigma)
        error_mnf[i] = error(X_hat_mnf, X)
        #X_hat_pca = do_pca(X_n, P[i])
        #error_pca[i] = error(X_hat_pca, X)
    print("P:         ", P)
    print("error_mnf: ", error_mnf)
    print("error_pca: ", error_pca)

    min_i = np.argmin(error_mnf)
    print("Smallest error for MNF found with P =", P[min_i])

    

#task3e()