from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import time
from task3_pca import do_pca
from task2 import kmeans_cluster, nasa_obpg, atmospheric_correction
from mnf import *

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
    x_pca, _ = do_pca(X, P)
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
        #x_cube = matrix_to_image_cube(X_n, [H,W,L])
        X_hat_mnf = spy_mnf(HICO_noisy, P[i], sigma_n)
        X_hat_mnf = image_cube_to_matrix(X_hat_mnf)
        error_mnf[i] = error(X_hat_mnf, X)
        _, X_hat_pca = do_pca(X_n, P[i])
        error_pca[i] = error(X_hat_pca, X)
    print("P:         ", P)
    print("error_mnf: ", error_mnf)
    print("error_pca: ", error_pca)

    min_i = np.argmin(error_mnf)
    print("Smallest error for MNF found with P =", P[min_i])

    rgb = (34, 25, 8)
    fignr = 5
    plt.figure(fignr)
    plt.subplot(121)
    plt.title("HICO noisy")
    spy.imshow(HICO_noisy, rgb, fignum=fignr)
    plt.subplot(122)
    plt.title("Denoised using MNF with P = " + str(P[min_i]))
    X_hat_mnf = spy_mnf(HICO_noisy, P[min_i], sigma_n)
    spy.imshow(X_hat_mnf, rgb, fignum=fignr)
    plt.savefig("fig/HICO_mnf.png")


def task3g():
    I = image_cube_to_matrix(HICO_original)
    P = 2
    colormap = "nipy_spectral"
    set_min_max = True

    _, X_hat_pca = do_pca(I, P)
    X_hat_pca = matrix_to_image_cube(X_hat_pca, [H,W,L])
    X = atmospheric_correction(HICO_original)
    X_hat_pca = atmospheric_correction(X_hat_pca)

    mask = np.load("land_mask.npy")
    plt.figure()
    for i in range(hico_wl.shape[0]):
        X_hat_pca[:,:,i] = X_hat_pca[:,:,i] * mask
        X[:,:,i] = X[:,:,i] * mask
    X_obpg_pca = nasa_obpg(X_hat_pca, "obpg_pca_masked_P"+str(P), set_min_max_values=set_min_max)
    X_obpg = nasa_obpg(X, "_", set_min_max_values=set_min_max)

    plt.subplot(121)
    plt.title("OBPG original HICO")
    plt.imshow(X_obpg, cmap=colormap)

    plt.subplot(122)
    plt.title("OBPG PCA compressed HICO, P = " + str(P))
    plt.imshow(X_obpg_pca, cmap=colormap)

    plt.savefig("fig/PCA_v_original_OBPG_P" + str(P) +".png")

def task3g_compare_Ps():
    I = image_cube_to_matrix(HICO_original)
    colormap = "nipy_spectral"
    set_min_max = True

    _, X_hat_pca2 = do_pca(I, 2)
    _, X_hat_pca3 = do_pca(I, 3)
    X_hat_pca2 = matrix_to_image_cube(X_hat_pca2, [H,W,L])
    X_hat_pca3 = matrix_to_image_cube(X_hat_pca3, [H,W,L])
    X_P2 = atmospheric_correction(X_hat_pca2)
    X_P3 = atmospheric_correction(X_hat_pca3)

    mask = np.load("land_mask.npy")
    plt.figure()
    for i in range(hico_wl.shape[0]):
        X_P2[:,:,i] = X_P2[:,:,i] * mask
        X_P3[:,:,i] = X_P3[:,:,i] * mask

    X_P2 = nasa_obpg(X_P2, "_", set_min_max_values=True)
    X_P3 = nasa_obpg(X_P3, "_", set_min_max_values=True)

    plt.subplot(121)
    plt.title("PCA + OBPG, P = 2")
    plt.imshow(X_P2, cmap=colormap)

    plt.subplot(122)
    plt.title("PCA + OBPG, P = 3")
    plt.imshow(X_P3, cmap=colormap)

    plt.savefig("fig/PCA_OBPG_2_3.png")

#task3e()
#task3g()
task3g_compare_Ps()
