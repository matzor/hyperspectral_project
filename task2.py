from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import pylab

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i
seawater_Rs   = M['seawater_Rs']

print("HICO_original.shape", HICO_original.shape)
print("HICO_noisy.shape", HICO_noisy.shape)
print("hico_wl.shape", hico_wl.shape)
print("seawater_Rs.shape", seawater_Rs.shape)

I = HICO_original

for i in range(1, 11): 
    m, c = spy.kmeans(I, i, 50)
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.title("Pixel colored by cluster, k = " + str(i))
    plt.imshow(m)
    plt.subplot(122)
    plt.title("K-cluster means")
    for j in range(c.shape[0]):
        pylab.plot(hico_wl[:,0], c[j])
    plt.xlabel("Wavelength [nm]")
    plt.savefig("fig/kmean/kmean_" + str(i) + ".png")
