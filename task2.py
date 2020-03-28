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

rgb = (34, 25, 8)
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

def kmeans_cluster(I, filename):
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
        plt.savefig("fig/kmean/" + filename + str(i) + ".png")

def nasa_obpg(I, filename):
    """ 
    Runs NASA OBPG algorithm with default parameters,
    saves result to file
    """
    # Parameters for the NASA OBPG algorithm 
    I_temp = np.empty_like(I)
    I_temp[:,:,:] = I[:,:,:]

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
    #I_temp[land_mask] = 0
    
    for x in range(xdim):
        for y in range(ydim):
            img[x, y] = a[0]
            for i in range(1, len(a)):
                img[x, y] += a[i] * np.log10(np.max(I_temp[x, y, i_blue]) / I_temp[x, y, i_green])
    img = 10 ** img
    #p_min = -2
    #p_max = 1
    plt.figure()
    plt.title(filename)
    #plt.imshow(img, vmax=p_max, vmin=p_min)
    plt.imshow(img)
    plt.savefig("fig/" + filename + ".png")
    


def calculate_atmospheric_scattering_coefficients(Img, points, Rrs, i_lambda_start, i_lambda_stop):
    """ 
    Calculates the a and b coefficients for estimating 
    reflectance from the surface through atmosphere
    Inputs:
        Img: HICO array
        point: reference points, ie. deep water or shallow water, 
            shape: [[x, y], [x, y]]
        Rrs: Calibrated remote sensing reflectance
            shape: np array 100 x 2
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

def atmospheric_correction(I):
    """ 
    Perform atmospheric correction by Empirical Line Method (ELM)
    Also prints out color changes between original and output image
    (debug info)
    Input:
        I: Image cube to be corrected
    Returns:
        img: corrected image cube
    """
    points = np.array([[20, 20], [100, 70]])      # Deep water, Shallow water
    R = 34
    G = 25
    B = 8

    i_start = get_index_of_wavelength(438)
    i_stop = get_index_of_wavelength(730)

    a, b = calculate_atmospheric_scattering_coefficients(I, 
        points, Rrs, i_start, i_stop)

    fig_n = 9
    plt.figure(fig_n)
    plt.subplot(121)
    plt.title("Original")
    spy.imshow(I, (34, 25, 8), fignum=fig_n)


    RGB_d_org = [I[20, 20, R], I[20, 20, G], I[20, 20, B]]
    RGB_s_org = [I[100, 70, R], I[100, 70, G], I[100, 70, B]]

    R_mean = np.mean(I[:,:,R])
    G_mean = np.mean(I[:,:,G])
    B_mean = np.mean(I[:,:,B])

    print("Deep water: ", RGB_d_org, "Shallow water: ", RGB_s_org)
    
    img = np.zeros_like(I)
    for i in range(i_start, i_stop):
        img[:,:,i] = (I[:,:,i] - b[i]) / a[i]

    plt.subplot(122)
    plt.title("Corrected")
    spy.imshow(I, (34, 25, 8), fignum=fig_n)
    plt.savefig("fig/pseudo_rgb_corrected.png")
    #plt.show()
    

    RGB_d_cor = [img[20, 20, R], img[20, 20, G], img[20, 20, B]]
    RGB_s_cor = [img[100, 70, R], img[100, 70, G], img[100, 70, B]]

    R_mean_cor = np.mean(img[:,:,R])
    G_mean_cor = np.mean(img[:,:,G])
    B_mean_cor = np.mean(img[:,:,B])

    print("Deep water: ", RGB_d_cor, "Shallow water: ", RGB_s_cor)

    print("Color ratio:")
    print("-"*80)
    print(R_mean_cor/R_mean, G_mean_cor/G_mean, B_mean_cor/B_mean)

    return img

def make_land_mask(I):
    """
    Creates a land mask, masking out land, showing only water
    Checks for peaks in pixel intensity after specific wavelength
    Assuming water has decreasing intensity for increasing wavelength.
    Also saves land mask as png. 
    Input:
        I: Image cube (hyperspectral, HICO)
    Returns:
        img: land mask numpy array
    """
    # Tweakable parameters: 
    max_peak = 0.55
    start_index = get_index_of_wavelength(700)
    stop_index = get_index_of_wavelength(730)
    k = 10
    m, c = spy.kmeans(I, k, 100)

    # Plotting the kmeans result for comparison
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.title("Pixel colored by cluster, k = " + str(k))
    plt.imshow(m)
    plt.subplot(122)
    plt.title("K-cluster means")
    legs = range(c.shape[0])
    for j in range(c.shape[0]):
        pylab.plot(hico_wl[:,0], c[j])
    plt.legend(legs)
    plt.xlabel("Wavelength [nm]")
    plt.savefig("fig/landmask_kmean.png")

    # Check mean values of k means after start index
    # Also want i=8 for some reason
    img = np.zeros_like(m)
    for i in range(c.shape[0]):
        print(np.mean(c[i, start_index:stop_index]))
        if np.mean(c[i, start_index:stop_index]) < max_peak or i==8:
            print("Found an acceptable class!")
            for x in range(m.shape[0]):
                for y in range(m.shape[1]):
                    if m[x,y] == i:
                        img[x,y] = 1
                        
    plt.figure()
    plt.imshow(img)
    plt.savefig("fig/landmask.png")
    np.save("land_mask", img)
    return img

def plot_masked_image(I):
    mask = np.load("land_mask.npy")

    fignr = 5
    plt.figure(fignr)
    plt.subplot(121)
    plt.title("Original RGB")
    spy.imshow(I, rgb, fignum=fignr)

    plt.subplot(122)
    plt.title("RGB with land mask")

    #I[land_mask] = 0

    for i in range(hico_wl.shape[0]):
        I[:,:,i] = I[:,:,i] * mask
        
    ### WHY DOES IT CHANGE COLORS???

    spy.imshow(I, rgb, fignum=fignr)
    #spy.imshow(I, rgb)

    plt.savefig("fig/rgb_masked.png")    

    nasa_obpg(I, "masked_obpg.png")


## doing the stuff 

#kmeans_cluster(I, "kmean_")

#nasa_obpg(I, "NASA OBPG, original")

I_cor = atmospheric_correction(I)

#nasa_obpg(I_cor, "NASA OBPG, corrected")

#kmeans_cluster(I_cor, "kmean_corrected")

#mask = make_land_mask(I_cor)

plot_masked_image(I_cor)
