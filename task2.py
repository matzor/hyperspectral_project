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
land_mask     = plt.imread('data/land_mask.png')[:,:,0] == 0

deep_water_Rrs = np.loadtxt('data/deep_water_Rrs.txt')
shallow_water_Rrs = np.loadtxt('data/shallow_water_Rrs.txt')

Rrs = np.empty((deep_water_Rrs.shape[0], 2))
Rrs[:,0] = deep_water_Rrs
Rrs[:,1] = shallow_water_Rrs

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
            #print("L ", i, hico_wl[i])
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

def nasa_obpg():
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
    I[land_mask] = 0
    
    for x in range(xdim):
        for y in range(ydim):
            img[x, y] = a[0]
            for i in range(1, len(a)):
                img[x, y] += a[i] * np.log10(np.max(I[x, y, i_blue]) / I[x, y, i_green])
    #img = 10 ** img
    plt.figure()
    plt.imshow(img)
    plt.savefig("fig/2b_nasa_cor.png")
    plt.show()


def calculate_atmospheric_scattering_coefficients(Img, points, Rrs, i_lambda_start, i_lambda_stop):
    """ 
    Calculates the a and b coefficients for estimating 
    reflectance from the surface through atmosphere
    Inputs:
        Img: HICO array
        point: reference points, ie. deep water or shallow water,
        (x, y) coordinates
        Rrs: Calibrated remote sensing reflectance
        i_lambda_start: index of first wavelength
        i_lambda_stop: index of last wavelength
    Returns:
        a, b
        array containing all a_i, b_i for all lambda_i
        a_i, b_i is zero for any lambda_i outside (i_lambda_start, i_lambda_stop)
    """

    length = i_lambda_stop - i_lambda_start
    # Get one pair (a, b) for each wavelength:
    X = np.zeros((Rrs.shape[0], 2))
    for i in range(length):
        a = np.array([  
                        [Rrs[i_lambda_start+i, 0],  1],
                        [Rrs[i_lambda_start+i, 1],  1]
                    ])
        b = np.array([  
                        [Img[points[0, 0], points[0, 1], i_lambda_start + i]],
                        [Img[points[1, 0], points[1, 1], i_lambda_start + i]]
                    ])
        #print("A: \n", a.shape)
        #print("B: \n", b.shape)
        x = np.linalg.solve(a, b)
        #x, _, _, _ = np.linalg.lstsq(a, b)
        #print("x: \n", x.shape)
        X[i_lambda_start + i, 0] = x[0, 0]
        X[i_lambda_start + i, 1] = x[1, 0]

    #print(X)
    return X[:, 0], X[:, 1]     #a, b



## Estimate the reflectance from the surface of the ocean

def run_task_c():
    points = np.array([[20, 20], [100, 70]])      # Deep water, Shallow water
    R = 34
    G = 25
    B = 8

    i_start = get_index_of_wavelength(438)
    i_stop = get_index_of_wavelength(730)

    a, b = calculate_atmospheric_scattering_coefficients(I, 
        points, Rrs, i_start, i_stop)

    plt.figure(1)
    plt.subplot(121)
    plt.title("Original")
    spy.imshow(I, (34, 25, 8), fignum=1)


    RGB_d_org = [I[20, 20, R], I[20, 20, G], I[20, 20, B]]
    RGB_s_org = [I[100, 70, R], I[100, 70, G], I[100, 70, B]]

    R_mean = np.mean(I[:,:,R])
    G_mean = np.mean(I[:,:,G])
    B_mean = np.mean(I[:,:,B])

    print("Deep water: ", RGB_d_org, "Shallow water: ", RGB_s_org)
    

    for i in range(i_start, i_stop):
        I[:,:,i] = (I[:,:,i] - b[i]) / a[i]

    plt.subplot(122)
    plt.title("Corrected")
    spy.imshow(I, (34, 25, 8), fignum=1)
    plt.savefig("fig/pseudo_rgb_corrected.png")
    plt.show()
    

    RGB_d_cor = [I[20, 20, R], I[20, 20, G], I[20, 20, B]]
    RGB_s_cor = [I[100, 70, R], I[100, 70, G], I[100, 70, B]]

    R_mean_cor = np.mean(I[:,:,R])
    G_mean_cor = np.mean(I[:,:,G])
    B_mean_cor = np.mean(I[:,:,B])

    print("Deep water: ", RGB_d_cor, "Shallow water: ", RGB_s_cor)

    print("Color ratio:")
    for i in range(len(RGB_d_cor)):
        print(i)
        print(RGB_d_cor[i] / RGB_d_org[i])
        print(RGB_s_cor[i] / RGB_s_org[i])
    
    print("-"*80)
    print(R_mean_cor/R_mean, G_mean_cor/G_mean, B_mean_cor/B_mean)

run_task_c()

nasa_obpg()