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


def harmonic(frame, sample_rate):
    M = np.round(0.016 * sample_rate) - 1
    R = np.correlate(frame, frame, mode='full')
    g = R[len(frame)-1]
    R = R[len(frame):-1]
    # estimate m0 (as the first zero crossing of R)
    [a, ] = np.nonzero(np.diff(np.sign(R)))
    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1
    Gamma = np.zeros((M), dtype=np.float64)
    CSum = np.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (np.sqrt((g * CSum[M:m0:-1])) + _eps)
    ZCR = zcr(Gamma)
    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = np.zeros((M), dtype=np.float64)
        else:
            HR = np.max(Gamma)
            blag = np.argmax(Gamma)
        # Get fundamental frequency:
        f0 = sample_rate / (blag + _eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0
    return (HR, f0)


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
    signal = signal / (2.0 ** 15)
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
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 13
    totalNumOfFeatures = numOfTimeSpectralFeatures + \
        nceps + numOfHarmonicFeatures + numOfChromaFeatures
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


def normalize(features):
    X = np.array([])
    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = np.vstack((X, f))
            count += 1
    MEAN = np.mean(X, axis=0)
    STD = np.std(X, axis=0)
    featuresNorm = []
    for f in features:
        ft = f.copy()
        for nSamples in range(f.shape[0]):
            ft[nSamples, :] = (ft[nSamples, :] - MEAN) / STD
        featuresNorm.append(ft)
    return (featuresNorm, MEAN, STD)
