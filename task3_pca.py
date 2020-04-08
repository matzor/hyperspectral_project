from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from task3 import image_cube_to_matrix, matrix_to_image_cube

pd.set_option('max_rows', 500)
pd.set_option('max_columns', 500)
np.set_printoptions(suppress=True)

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i


# Based on
# https://www.machinelearningplus.com/machine-learning/principal-components-analysis-pca-better-explained/


def task3_b_test():
    # Creating the dataset (random data)
    data = np.random.rand(100, 1000)
    # Want columns to be x
    x = data.reshape(data.shape[1], data.shape[0]) 
    print("Original data shape: ", x.shape)


    # PCA
    pca = PCA()
    df_pca = pca.fit_transform(X = x)

    # Store as dataframe and print
    df_pca = pd.DataFrame(df_pca)
    print("PCA dataframe shape: ", df_pca.shape)     # Same shape as original data
    df_pca.round(2).head()

    # Principal Components Weights
    df_pca_weights = pd.DataFrame(pca.components_)
    print("PCA weights shape: ", df_pca_weights.shape)
    df_pca_weights.head()

    print(pca.explained_variance_ratio_.round(2)[:])