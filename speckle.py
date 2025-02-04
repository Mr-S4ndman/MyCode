import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy
from scipy import ndimage
from PIL import Image
from photutils.detection import find_peaks
import json

if __name__ == "__main__":

    data = fits.open('speckledata.fits')[2].data
    mean = np.mean(data, axis=0)
    
    mean1 = Image.fromarray(mean).resize((512, 512))
    plt.figure()
    plt.imsave("mean.png", np.abs(mean1), cmap='gray')
    fourier = np.mean(np.abs(np.fft.fft2(data))**2, axis=0)
    E = np.fft.fftshift(fourier)
    
    E1 = Image.fromarray(E).resize((512, 512))
    plt.figure()
    plt.imsave("fourier.png", np.abs(E1), vmax=1e+9, cmap='gray')
    
    maskcirc = np.full_like(E, 1, dtype = np.float32())
    for i in range(len(E)):
        for j in range(len(E)):
            if (i-100)**2+(j-100)**2 <= 2500:
                maskcirc[i][j]=0
    
    maskedE = E * maskcirc
    
    clearE = E - np.mean(maskedE)
    
    
    sumrotE = np.zeros((200, 200))
    for i in range(360):
        sumrotE = sumrotE + scipy.ndimage.rotate(E, i, reshape=False)
    sumrotE = sumrotE/360
    
    sumrot1 = Image.fromarray(sumrotE).resize((512, 512))
    plt.figure()
    plt.imsave("rotaver.png", np.abs(sumrot1), vmax=1e+9, cmap='gray')
    normE = clearE/sumrotE 
    antimaskcirc = np.full_like(E, 0, dtype = np.float32())
    for i in range(len(E)):
        for j in range(len(E)):
            if (i-100)**2+(j-100)**2 <= 2500:
                antimaskcirc[i][j]=1
    antimaskedE = normE * antimaskcirc
    
    ifourier = abs(np.fft.ifft2(antimaskedE))
    ifourier = np.fft.fftshift(ifourier)
    ifourier1 = Image.fromarray(ifourier).resize((512, 512))
    plt.figure()
    plt.imsave("binary.png", np.abs(ifourier1), cmap='gray')
    threshold = np.mean(np.abs(ifourier)) + (1.44 * np.std(np.abs(ifourier)))
    peaks = find_peaks(np.abs(ifourier), threshold, box_size=11)
    xpositions = np.transpose((peaks['x_peak']))
    ypositions = np.transpose((peaks['y_peak']))
    radiase = ((xpositions[0]-xpositions[2])**2+(ypositions[0]-ypositions[2])**2)**0.5/2*0.0206
    dictionary = {"distance:": radiase}
    with open('binary.json', 'w') as f:
        json.dump(dictionary, f)
    
    
