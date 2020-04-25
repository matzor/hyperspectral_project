import numpy as np
from numpy.linalg import eig

def pca(X):
    # Standardize each column
    print("X:, ", X.shape)
    X_standard = np.empty_like(X)
    for col in range(X.shape[1]):
        X_standard[:, col] = X[:,col] - np.mean(X[:,col])
    print("X_standard shape:, ", X_standard.shape)
    
    # Compute covariance
    X_cov = np.cov(X_standard)
    
    # Eigenvalues and eigenvectors
    eigvals, eigvecs = eig(X_cov)
    print("Eigvecs:", eigvecs.shape)

    # Derive principal components
    # by taking dot product of X
    # and eigenvectors
    X_pca = np.dot(X_standard.T, eigvecs)

    return X_pca.T