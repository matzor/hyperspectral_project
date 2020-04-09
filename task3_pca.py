from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

#from task3 import image_cube_to_matrix, matrix_to_image_cube

pd.set_option('max_rows', 500)
pd.set_option('max_columns', 500)
np.set_printoptions(suppress=True)


# Based on
# https://www.machinelearningplus.com/machine-learning/principal-components-analysis-pca-better-explained/

def encircle(x, y, ax=None, **kw):
    """ 
    Encircles points within the cluster, from machinelearningplus """
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)



def do_pca(datamatrix, num_components):
    """ 
    Perform Principal Component Analysis on the dataset (on matrix form) 

    Input:
        datamatrix: np.array X with L rows and N = WH columns
    Returns:
        df_pca: pca output, p x n
    """
    # Preparing the dataset
    x = datamatrix.T
    #x = datamatrix
    print("Original data shape: ", x.shape)

    # PCA
    pca = PCA(num_components)
    np_pca = pca.fit_transform(X = x)

    # Store as dataframe
    df_pca = pd.DataFrame(np_pca)
    print("PCA dataframe shape: ", df_pca.shape)     # Same shape as original data
    #df_pca.round(2).head()

    # Principal Components Weights (Eigenvectors)
    df_pca_weights = pd.DataFrame(pca.components_)
    print("PCA weights shape: ", df_pca_weights.shape)
    #df_pca_weights.head()

    # printing precentage of variance of each PC
    print(pca.explained_variance_ratio_.round(2)[:])

    # Visualizing variance of each principle component
    variance_exp_cumsum = pca.explained_variance_ratio_.cumsum().round(2)
    fig, axes = plt.subplots(1, 1, figsize=(16, 7), dpi=100)
    plt.plot(variance_exp_cumsum, color='firebrick')
    plt.title("Variance by Principal Component", fontsize=22)
    plt.xlabel("# of PCs", fontsize=16)
    plt.savefig("fig/pca/variance_cumsum.png")
    #plt.show()

    # getting original data back
    df_orig = pca.inverse_transform(df_pca)
    #pd.DataFrame(df_orig).round().head()
    #print(df_orig.shape)
    #np_pca = df_orig
    #np_pca = np_pca.T
    return np_pca.T
    


def task3_b_test():
    """ 
    Tests the output sizes of PCA
    Don't use this, rather use do_pca() with random input
    of correct size.  """
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