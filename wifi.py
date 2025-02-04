import argparse
import numpy as np
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument('input',  metavar='FILENAME', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    name = args.input
    data = pd.read_csv(name, names=['txt']).txt

    code1 = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1], dtype=np.int8)
    code = np.repeat(code1, 5)
    dc = np.convolve(data, code[::-1], mode = 'full')
    ms = np.mean(dc)
    std = np.std(dc)
    peaks = dc
    peaks[np.abs(dc) < 2 * std] = 0
    peaks[dc > (2 * std)] = 1
    peaks[dc < -2 * std] = -1
    for i in range (len(peaks)):
        if dc[i] == dc[i-1]:
            dc[i-1] = 0
    bin = peaks[peaks !=0]
    bin[bin == -1] = 0
    x = np.split(bin, 8)
    bits = np.packbits(np.asarray(x, dtype=np.uint8))
    w = bits.tobytes()
    word = w.decode(encoding = "ascii")
    outw = {"message": word}
    with open('wifi.json', 'w') as f:
        json.dump(outw, f)