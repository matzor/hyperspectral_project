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
    """ 
    Input: Approximate wavelength in HICO array, in nm

    Output: Index of closest wavelength in HICO array
    """
    from math import isclose
    for i in range(hico_wl.shape[0]):
        if isclose(hico_wl[i], wavelength, abs_tol=5.7/2):
            print("L ", i, hico_wl[i])
            return i

def kmeans_cluster():
    """ 
    Runs K-means clustering on HICO array with 1 - 10 number
    of classes, and saves result to file
    """
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
    """ 
    Runs NASA OBPG algorithm with default parameters
    """
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

def OLD_calculate_atmospheric_scattering_coefficients(Img, point, Rrs):
    """ 
    Calculates the a and b coefficients for estimating 
    reflectance from the surface through atmosphere
    Inputs:
        Img: HICO array
        point: reference point, ie. deep water or shallow water,
        (x, y) coordinates
        Rrs: Calibrated remote sensing reflectance
        Wavelength: 
    Returns:
        a, b
    """
    ones = np.ones(Rrs.shape[0])
    A = np.empty((Rrs.shape[0], 2))
    A[:,0] = Rrs
    A[:,1] = ones
    B = Img[point[0], point[1], :]
    B = B.reshape((100, 1))
    x, _, _, _ = np.linalg.lstsq(A, B)
    print(x)
    return x[0, 0], x[1, 0]

def calculate_atmospheric_scattering_coefficients(Img, point, Rrs, i_lambda_start, i_lambda_stop):
    """ 
    Calculates the a and b coefficients for estimating 
    reflectance from the surface through atmosphere
    Inputs:
        Img: HICO array
        point: reference point, ie. deep water or shallow water,
        (x, y) coordinates
        Rrs: Calibrated remote sensing reflectance
        i_lambda_start: index of first wavelength
        i_lambda_stop: index of last wavelength
    Returns:
        a, b
    """
    # 
    # TODO: Shouldn't we get an a and b for each wavelength?
    # Not just one for the whole span?
    # 

    length = i_lambda_stop - i_lambda_start
    ones = np.ones(length)
    A = np.empty((length, 2))
    A[:, 0] = Rrs[i_lambda_start:i_lambda_stop]
    A[:, 1] = ones
    B = Img[point[0], point[1], i_lambda_start:i_lambda_stop]

    #print("A: \n", A.shape)
    #print("B: \n", B.shape)

    X, _, _, _ = np.linalg.lstsq(A, B)

    
    # Get one pair (a, b) for each wavelength:
    #X = np.empty((length, 2))

    """
    for i in range(length):
        a = A[i, :]#.reshape((2,1))
        b = B[i]#.reshape((1, 1))
        print("A: \n", a)
        print("B: \n", b)
        X[i], _, _, _ = np.linalg.solve(a, b) 
    """
    print(X)
    return X[0], X[1]



## Estimate the reflectance from the surface of the ocean

points = [[20, 20], [100, 70]]      # Deep water, Shallow water

i_start = get_index_of_wavelength(438)
i_stop = get_index_of_wavelength(730)

a_deep, b_deep = calculate_atmospheric_scattering_coefficients(I, 
    points[0], deep_water_Rrs, i_start, i_stop)
a_shallow, b_shallow = calculate_atmospheric_scattering_coefficients(I, 
    points[1], shallow_water_Rrs, i_start, i_stop)