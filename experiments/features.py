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

    frame_len = 2048
    n_fft = 1024
    window = hamming(2048)
    [fbank, freqs] = mfcc.init(22050.0, n_fft)
    clipfeats = list()
    s_prev = None
    for i in xrange(0, len(clip)-frame_len+1, frame_len):
        frame = clip[i:i+frame_len] * window
        s = np.abs(np.fft.rfft(frame))[:n_fft] / float(n_fft)
        [centroid, spread] = spectral.centroid_and_spread(s, 22050)
        entropy = spectral.entropy(s)
        flux = spectral.flux(s, s_prev) if not s_prev is None else 0.0
        rolloff = spectral.rolloff(s, 0.90)
        mfccs = mfcc.filter(s, fbank, 13)
        spectrals = [centroid, spread, entropy, flux, rolloff]
        framefeats = np.concatenate(([zcr(s), energy(s)], spectrals, mfccs))
        clipfeats.append(framefeats)
        s_prev = s
    return np.array(clipfeats)


def plot_feats(feats, ax):
    pos = np.arange(feats.shape[1])
    ax.violinplot(feats, pos, points=400, vert=False, widths=0.9,
            showmeans=False, showextrema=True, showmedians=True)
    # highlight the MFCC features
    ax.axhspan(8-0.2,20+0.2, facecolor='red', alpha=0.15)
    # highlight the chroma features
    ax.axhspan(21-0.2,32+0.2, facecolor='blue', alpha=0.15)


def run(clips):
    # Measure the effectiveness of different features for music similarity
    # calculation.
    # Compute the distribution of feature values across the track list.
    # Plot a histogram for each feature.

    orig_feats = list()
    for t, clip_A, feats_A, clip_B, feats_B in clips:
        orig_feats.append(feats_A)
        orig_feats.append(feats_B)
    orig_feats = np.concatenate(orig_feats, axis=0)

    norm_feats = (orig_feats - orig_feats.mean(axis=0)) / orig_feats.std(axis=0)

    # Display distributions of features with a violin plot.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    plot_feats(orig_feats, axes[0])
    axes[0].set_title("tracks db features")

    plot_feats(norm_feats, axes[1])
    axes[1].set_title("normalized features")

    plt.savefig("features.png", dpi=96, bbox_inches='tight')

