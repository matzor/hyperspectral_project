from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i
seawater_Rs   = M['seawater_Rs']

print(HICO_original.shape)
print(HICO_noisy.shape)
print(hico_wl.shape)
print(seawater_Rs.shape)

# To convert a hyperspectral image cube I to matrix form X:
I = HICO_original
[H,W,L] = I.shape
X = np.reshape(I, [H*W,L])
X = X.T

# To convert a matrix X back into a hyperspectral image cube:
I = np.reshape(X.T, [H,W,L])

# Plot a single spectral band
plt.imshow(I[:,:,30])
plt.show()

# Note that quite a few libraries assume a matrix layout where
# each row is a spectral vector, rather than each column as in
# equation 2 of the assignment text. Read the documentation of
# those libraries carefully.
