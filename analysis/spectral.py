import numpy as np


def centroid_and_spread(X, sample_rate):
    ind = (np.arange(1, len(X) + 1)) * (sample_rate/(2.0 * len(X)))
    Xt = X.copy()
    Xt = Xt / Xt.max()
    eps = np.finfo(X.dtype).eps
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps
    centroid = (NUM / DEN)
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)
    # Normalize:
    centroid /= (sample_rate / 2.0)
    spread /= (sample_rate / 2.0)
    return (centroid, spread)


def entropy(X, numOfShortBlocks=10):
    L = len(X)                         # number of frame samples
    Eol = np.sum(X ** 2)            # total spectral energy
    subWinLength = int(np.floor(L / numOfShortBlocks)
                       )   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]
    # define sub-frames (using matrix reshape)
    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()
    # compute spectral sub-energies
    eps = np.finfo(X.dtype).eps
    s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)
    # compute spectral entropy
    En = -np.sum(s*np.log2(s + eps))
    return En


def flux(X, Xprev):
    # compute the spectral flux as the sum of square distances:
    eps = np.finfo(X.dtype).eps
    sumX = np.sum(X + eps)
    sumPrevX = np.sum(Xprev + eps)
    F = np.sum((X / sumX - Xprev/sumPrevX) ** 2)
    return F


def rolloff(X, c):
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Find the spectral rolloff as the frequency position where the respective
    # spectral energy is equal to c*totalEnergy
    eps = np.finfo(X.dtype).eps
    CumSum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(CumSum > Thres)
    mC = np.float64(a[0]) / (float(fftLength)) if len(a) > 0 else 0.0
    return (mC)
