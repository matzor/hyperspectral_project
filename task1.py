from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i
seawater_Rs   = M['seawater_Rs']

print("HICO_original.shape", HICO_original.shape)
print("HICO_noisy.shape", HICO_noisy.shape)
print("hico_wl.shape", hico_wl.shape)
print("seawater_Rs.shape", seawater_Rs.shape)

#print("Wavelengths: \n", hico_wl)

#for i in range(1, hico_wl.shape[0]):
    #print("Delta L", i, ": ", hico_wl[i, 0] - hico_wl[i-1, 0])
    
#for i in range(hico_wl.shape[0]):
    #print("L ", i, hico_wl[i])
    #Blue: 450nm, i = 8
    #Green: 550nm, i = 25
    #Red: 600nm, i = 34

# To convert a hyperspectral image cube I to matrix form X:
I = HICO_original
[H,W,L] = I.shape
X = np.reshape(I, [H*W,L])
X = X.T

# To convert a matrix X back into a hyperspectral image cube:
I = np.reshape(X.T, [H,W,L])
#I = I / 255.0

# Plot a single spectral band
#plt.imshow(I[:,:,30])
#plt.show()

# Plotting pseudo RGB
""" I_rgb = np.empty([H, W, 3])
I_rgb[:,:,0] = I[:,:,7] 
I_rgb[:,:,1] = I[:,:,23] 
I_rgb[:,:,2] = I[:,:,30] 
print(I_rgb.shape)
plt.imshow(I_rgb[:,:,:])

plt.savefig("fig/pseudo_rgb.png")
plt.show() """

# Pseudo RGB
view = spy.imshow(I, (34, 25, 8))
plt.savefig("fig/pseudo_rgb.png")
#spy.save_rgb('rgb.jpg', I, [34, 25, 8])

# RGB with points
plt.figure(10)
plt.subplot(121, aspect='auto')
plt.title("Spectrum of selected points")
points = np.array([[20, 100, 400], [20, 70, 30]]) # [[x0, x1, x2], [y0, y1, y2]]
colors = ['g', 'r', 'y']
legends = ["Deep water", "Shallow water", "Vegetation"]
view = spy.imshow(I, (34, 25, 8), fignum=10)
plt.scatter(points[0,:],points[1,:], c=colors, marker='x', label=legends)

#plt.savefig("fig/pseudo_rgb_points.png")

#plt.figure()
plt.subplot(122, aspect='auto')
for i in range(points.shape[0]+1):
    plt.plot(hico_wl[:,0], I[points[1,i], points[0,i], :], c=colors[i])
plt.legend(legends)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Power")

#plt.savefig("fig/pseudo_rgb_points_spectra.png")

plt.savefig("fig/pseudo_rgb_points.png")

plt.show()





# Note that quite a few libraries assume a matrix layout where
# each row is a spectral vector, rather than each column as in
# equation 2 of the assignment text. Read the documentation of
# those libraries carefully.

input()