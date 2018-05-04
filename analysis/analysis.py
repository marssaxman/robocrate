import numpy as np
import scipy.fftpack
from scipy.fftpack import fft
import mfcc
import chroma
import spectral


_eps = 0.00000001


def zcr(frame):
    count = len(frame)
    count_zeros = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(count_zeros) / np.float64(count-1.0))


def energy(frame):
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, numOfShortBlocks=10):
    Eol = np.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(np.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
        frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(
        subWinLength, numOfShortBlocks, order='F').copy()
    # Compute normalized sub-frame energies:
    s = np.sum(subWindows ** 2, axis=0) / (Eol + _eps)
    # Compute entropy of the normalized sub-frame energies:
    Entropy = -np.sum(s * np.log2(s + _eps))
    return Entropy


def extract(signal, sample_rate, window=1.0, step=0.5):
    # 0: zero-crossing rate
    # 1: energy
    # 2: energy entropy
    # 3: spectral centroid
    # 4: spectral spread
    # 5: spectral entropy
    # 6: spectral flux
    # 7: spectral rolloff
    # 8..21: MFCC bands
    # 22..35: chroma bands
    Win = int(window * sample_rate)
    Step = int(step * sample_rate)
    # Signal normalization
    signal = np.double(signal)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2
    # compute the triangular filter banks used in the mfcc calculation
    [fbank, freqs] = mfcc.init(sample_rate, nFFT)
    nChroma, nFreqsPerChroma = chroma.init(nFFT, sample_rate)
    numOfTimeSpectralFeatures = 8
    nceps = 13
    numOfChromaFeatures = 13
    totalNumOfFeatures = numOfTimeSpectralFeatures + \
        nceps + numOfChromaFeatures
    stFeatures = []
    # for each short-term window until the end of signal
    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            # keep previous fft mag (used in spectral flux)
            Xprev = X.copy()
        curFV = np.zeros((totalNumOfFeatures, 1))
        curFV[0] = zcr(x)                              # zero crossing rate
        curFV[1] = energy(x)                           # short-term energy
        # short-term entropy of energy
        curFV[2] = energy_entropy(x)
        [curFV[3], curFV[4]] = spectral.centroid_and_spread(
            X, sample_rate)    # spectral centroid and spread
        curFV[5] = spectral.entropy(X)                  # spectral entropy
        curFV[6] = spectral.flux(X, Xprev)              # spectral flux
        curFV[7] = spectral.rolloff(X, 0.90)        # spectral rolloff
        curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures +
              nceps, 0] = mfcc.filter(X, fbank, nceps).copy()    # MFCCs
        chromaF = chroma.features(X, sample_rate, nChroma, nFreqsPerChroma)
        curFV[numOfTimeSpectralFeatures + nceps: numOfTimeSpectralFeatures +
              nceps + numOfChromaFeatures - 1] = chromaF
        curFV[numOfTimeSpectralFeatures + nceps +
              numOfChromaFeatures - 1] = chromaF.std()
        stFeatures.append(curFV)
        Xprev = X.copy()
    stFeatures = np.concatenate(stFeatures, 1)
    return stFeatures

