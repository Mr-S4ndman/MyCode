import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, irfft
import matplotlib.ticker as ticker
import os
import re
from scipy.io.wavfile import read

filename = os.listdir(r"C:\Users\sasha\OneDrive\Рабочий стол\matches2")
os.chdir(r"C:\Users\sasha\OneDrive\Рабочий стол\matches2")

def get_numbers_from_filename(filename):
    return int(re.search(r'\d+', filename).group(0))
def filtemax(array, N):
    d = np.array_split(array, N)
    for i in range(len(d)):
        d[i]=d[i][np.argmax(d[i])]
    return d
def filtemean(array, N):
    d = np.array_split(array, N)
    for i in range(len(d)):
        d[i]=np.mean(d[i])
    return d
def get_fourier_from_wav(sound, size, down, rate):
    N = len(sound)  
    soundf = fft(sound)
    xf = fftfreq(N, 1/rate)
    soundff = soundf[np.abs(xf)<size]
    xff = xf[np.abs(xf)<size]
    soundff = soundff[xff>down]
    xff = xff[xff>down]
    soundff = soundff/np.max(np.abs(soundff))
    return xff, np.abs(soundff)
def SplitByNumber(array, N):
    lenght = len(array)
    parts = lenght // N
    return np.array_split(array, parts)
N = 6
freq = 44100
lenght = len(filename)
frame = pd.DataFrame()
for i in range(lenght):
    data = read(filename[i])
    rate = data[0]
    datasize = len(data[1])
    for j in range(datasize//(freq*N)):
        sound = SplitByNumber(data[1], N * freq)[j]
        a,b = get_fourier_from_wav(sound = sound, size = 3020, down = 20, rate = freq)
        d = filtemean(b,300)
        frame[i*(datasize//(freq*N))+j]=np.hstack([get_numbers_from_filename(filename[i]),d])
        
frame.to_csv(r'C:\Users\sasha\.spyder-py3\тест1.csv', index=False)

