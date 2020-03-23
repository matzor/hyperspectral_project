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

deep_water_Rrs = np.loadtxt('data/deep_water_Rrs.txt')
shallow_water_Rrs = np.loadtxt('data/shallow_water_Rrs.txt')

print("HICO_original.shape", HICO_original.shape)
print("HICO_noisy.shape", HICO_noisy.shape)
print("hico_wl.shape", hico_wl.shape)
print("seawater_Rs.shape", seawater_Rs.shape)
print("Rrs.shape", deep_water_Rrs.shape)

I = HICO_original

def get_index_of_wavelength(wavelength):
    from math import isclose
    for i in range(hico_wl.shape[0]):
        if isclose(hico_wl[i], wavelength, abs_tol=5.7/2):
            print("L ", i, hico_wl[i])
            return i

def kmeans_cluster():
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

def nasa_obpg(Img):
    # Parameters for the NASA OBPG algorithm 
    a = [0.3272, -2.9940, 2.7218, -1.2259, -0.5683]
    l_green = 555
    l_blue = [443, 490, 510]

    i_green = get_index_of_wavelength(l_green)
    i_blue = []
    for i in range(len(l_blue)):
        i_blue.append(get_index_of_wavelength(l_blue[i]))

    xdim = HICO_original.shape[0]
    ydim = HICO_original.shape[1]

    img = np.empty((xdim, ydim))
    
    for x in range(xdim):
        for y in range(ydim):
            img[x, y] = a[0]
            for i in range(1, len(a)):
                img[x, y] += a[i] * np.log10(np.max(I[x, y, i_blue]) / I[x, y, i_green])
    #img = 10 ** img
    plt.imshow(img)
    plt.savefig("fig/2b_nasa.png")
    plt.show()

def calculate_atmospheric_scattering_coefficients(Img, point, Rrs):
    ones = np.ones(Rrs.shape[0])
    A = np.empty((Rrs.shape[0], 2))
    A[:,0] = Rrs
    A[:,1] = ones
    B = Img[point[0], point[1], :]
    B = B.reshape((100, 1))
    print("A.shape:\n", A.shape)
    print("B.shape:\n", B.shape)
    a = np.linalg.solve(A, B)
    print(a)
    return a

print(calculate_atmospheric_scattering_coefficients(I, [20, 20], deep_water_Rrs))