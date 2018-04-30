import numpy as np


def init(nfft, sample_rate):
    freqs = np.array([((f + 1) * sample_rate) / (2 * nfft)
                      for f in range(nfft)])
    Cp = 27.50
    nChroma = np.round(12.0 * np.log2(freqs / Cp)).astype(int)
    nFreqsPerChroma = np.zeros((nChroma.shape[0], ))
    uChroma = np.unique(nChroma)
    for u in uChroma:
        idx = np.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    return nChroma, nFreqsPerChroma


def features(X, sample_rate, nChroma, nFreqsPerChroma):
    spec = X**2
    if nChroma.max() < nChroma.shape[0]:
        C = np.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:
        I = np.nonzero(nChroma > nChroma.shape[0])[0][0]
        C = np.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec
        C /= nFreqsPerChroma
    finalC = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(C2.shape[0]/12, 12)
    finalC = np.matrix(np.sum(C2, axis=0)).T
    finalC /= spec.sum()
    return finalC
