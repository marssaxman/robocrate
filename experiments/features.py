import os
import os.path
import numpy as np
from analysis import mfcc, spectral

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _caption(track):
    if track.title and track.artist:
        return "%s - %s" % (track.artist, track.title)
    if track.title:
        return track.title
    return os.path.splitext(os.path.basename(track.source))[0]


def statify_feats(feats):
    # There is no measurable difference between stacking the feature values
    # horizontally or vertically.
    return np.vstack((
        np.mean(feats, axis=0),
        np.std(feats, axis=0)
    ))


def plot_features(track, feats_A, feats_B):
    fig = plt.figure(1, figsize=(1024/96, 1280/96), dpi=96)
    plt.set_cmap('hot')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1,
                           width_ratios=[12, 1], wspace=0.1)
    axA = plt.subplot(gs[0, 0])
    axA.matshow(np.clip(feats_A.T, -3.0, 3.0), vmin=-3.0, vmax=3.0)
    axA.set_xlim(0, feats_A.shape[0])
    axA.set_ylim(feats_A.shape[1], 0)
    axA.autoscale(False)

    axAstat = plt.subplot(gs[0, 1], sharey=axA)
    axAstat.matshow(statify_feats(feats_A).T, vmin=-3.0, vmax=3.0)

    axB = plt.subplot(gs[1, 0], sharex=axA)
    axB.matshow(np.clip(feats_B.T, -3.0, 3.0), vmin=-3.0, vmax=3.0)
    axB.set_xlim(0, feats_B.shape[0])
    axB.set_ylim(feats_B.shape[1], 0)
    axB.autoscale(False)

    axBstat = plt.subplot(gs[1, 1], sharey=axB)
    axBstat.matshow(statify_feats(feats_B).T, vmin=-3.0, vmax=3.0)

    plt.savefig(_caption(track)+'_feats.png', dpi=96, bbox_inches='tight')



def altfeats(clip):
    def hamming(N):
        # improved hamming window: original implementation used 0.54, 0.46
        i = np.arange(N).astype(np.float)
        return 0.53836 - (0.46164 * np.cos(np.pi * 2.0 * i / (N-1)))

    def zcr(frame):
        count = len(frame)
        count_zeros = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        return (np.float64(count_zeros) / np.float64(count-1.0))

    def energy(frame):
        return np.sum(frame ** 2) / np.float64(len(frame))

    window = hamming(2048)
    [fbank, freqs] = mfcc.init(22050, 1025)
    clipfeats = list()
    s_prev = np.zeros(1025)
    for i in xrange(0, len(clip)-2047, 2048):
        frame = clip[i:i+2048] * window
        s = np.abs(np.fft.rfft(frame))
        s /= len(s)
        mfccs = mfcc.filter(s, fbank, 13)
        framefeats = mfccs
        [centroid, spread] = spectral.centroid_and_spread(s, 22050)
        entropy = spectral.entropy(s)
        flux = spectral.flux(s, s_prev)
        s_prev = s
        rolloff = spectral.rolloff(s, 0.90)
        spectrals = [centroid, spread, entropy, flux, rolloff]
        framefeats = np.concatenate(([zcr(s), energy(s)], spectrals, mfccs))
        clipfeats.append(framefeats)
    return np.array(clipfeats)


def run(clips):
    # the subject of this experiment is a little unclear
    # we want to measure the effectiveness of different groups of features
    # for music similarity identification
    pass

