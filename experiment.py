import os
import os.path
import sys
import library
import analysis
import audiofile
import scipy.spatial
import scipy.stats
import numpy as np
import random
from samplerate import resample
import argparse
import wave
import struct

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


def _path(*args):
    if len(args) > 1:
        track, suffix = args
        name = _caption(track) + suffix
    else:
        name = args[0]
    science_dir = os.path.join(os.getcwd(), "science")
    if not os.path.isdir(science_dir):
        os.makedirs(science_dir)
    return os.path.join(science_dir, name)


def calc_clips(track, plot=False):
    # Load audio file
    signal, samplerate = audiofile.read(track.source)
    # Mix to mono
    if hasattr(signal, 'ndim') and signal.ndim > 1:
        signal = signal.mean(axis=1).astype(np.float)
    # Resample down to 22050 Hz
    if samplerate > 22050.0:
        signal = resample(signal, 22050.0 / samplerate, 'sinc_fastest')
        samplerate = 22050.0
    # Normalize to -1..1
    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))
    # Extract some features and normalize them around their means.
    # We must transpose the features list because the extractor returns rows
    # of frames, and we want rows of features.
    features = analysis.extract(signal, samplerate, window=1.0, step=0.5).T
    orig_features = features
    features -= np.mean(features, axis=0)
    features /= np.std(features, axis=0)
    # Compute self-similarity, normalize to 0..1
    pairwise_dist = scipy.spatial.distance.pdist(features, 'cosine')
    sim_matrix = 1.0 - scipy.spatial.distance.squareform(pairwise_dist)
    # Score each half-second step for overall similarity.
    score = np.median(sim_matrix, axis=0)
    score -= np.min(score)
    score /= np.max(score)
    best_score = np.argmax(score)

    # Score each half-second step for suitability as the start of a 30-second
    # window - what is the average score for each such window?
    startscore = np.zeros(len(score)-60)
    for i in range(len(startscore)):
        startscore[i] = np.mean(score[i:i+60])
    best_starts = np.argsort(startscore)[::-1]
    best_A = best_starts[0]
    best_B = np.extract(np.abs(best_starts - best_A) >= 60.0, best_starts)[0]

    clip_A = signal[int(best_A * 0.5 * samplerate):][:int(samplerate * 30)]
    clip_B = signal[int(best_B * 0.5 * samplerate):][:int(samplerate * 30)]
    feats_A = orig_features[best_A:best_A + 60]
    feats_B = orig_features[best_B:best_B + 60]

    if plot:
        # Plot this stuff out so we can see how we're doing.
        fig = plt.figure(1, figsize=(1024/96,1280/96), dpi=96)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 12], hspace=0.1)
        plt.set_cmap('hot')
        axMatrix = plt.subplot(gs[2])
        axMatrix.set_aspect(1.)
        axScore = plt.subplot(gs[0], sharex=axMatrix)
        axStart = plt.subplot(gs[1], sharex=axMatrix)

        axScore.matshow(np.tile(score, (36, 1)))
        axScore.axis('off')
        axScore.axvline(best_score)
        axScore.text(best_score, 0, "Max")
        axScore.axvspan(best_A, best_A+60, color='blue', alpha=0.4)
        axScore.text(best_A, 0, "A")
        axScore.axvspan(best_B, best_B+60, color='blue', alpha=0.4)
        axScore.text(best_B, 0, "B")

        axStart.plot(startscore)
        axStart.set_ylim([0,1])
        axStart.get_xaxis().set_visible(False)
        axStart.axvline(best_A)
        axStart.text(best_A, 0, "A")
        axStart.axvline(best_B)
        axStart.text(best_B, 0, "B")

        axMatrix.matshow(sim_matrix)
        axMatrix.axis('off')

        plt.savefig(_path(track, '.png'), dpi=96, bbox_inches='tight')

    return ((clip_A, feats_A), (clip_B, feats_B))


def get_clips(t):
    try:
        feats_A = np.load(_path(t, "_A.npy"))
        feats_B = np.load(_path(t, "_B.npy"))
        clip_A = audiofile.read(_path(t, "_A.wav"))
        clip_B = audiofile.read(_path(t, "_B.wav"))
    except:
        (clip_A, feats_A), (clip_B, feats_B) = calc_clips(t)
        writewav(_path(t, "_A.wav"), clip_A)
        writewav(_path(t, "_B.wav"), clip_B)
        np.save(_path(t, "_A.npy"), feats_A)
        np.save(_path(t, "_B.npy"), feats_B)
    return (clip_A, feats_A), (clip_B, feats_B)


def writewav(path, clip, samplerate=22050):
    wf = wave.open(path, 'wb')
    if wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        samples = (clip * np.iinfo(np.int16).max).astype('<i2')
        wf.writeframesraw(samples.tobytes())
        wf.writeframes('')
        wf.close()


def statify_feats(feats):
    # There is no measurable difference between stacking the feature values
    # horizontally or vertically.
    return np.vstack((
        np.mean(feats, axis=0),
        np.std(feats, axis=0)
    ))


def plot_features(track, feats_A, feats_B):
    fig = plt.figure(1, figsize=(1024/96,1280/96), dpi=96)
    plt.set_cmap('hot')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1,
        width_ratios=[12, 1], wspace=0.1)
    axA = plt.subplot(gs[0,0])
    axA.matshow(np.clip(feats_A.T, -3.0, 3.0), vmin=-3.0, vmax=3.0)
    axA.set_xlim(0,feats_A.shape[0])
    axA.set_ylim(feats_A.shape[1], 0)
    axA.autoscale(False)

    axAstat = plt.subplot(gs[0,1], sharey=axA)
    axAstat.matshow(statify_feats(feats_A).T, vmin=-3.0, vmax=3.0)

    axB = plt.subplot(gs[1,0], sharex=axA)
    axB.matshow(np.clip(feats_B.T, -3.0, 3.0), vmin=-3.0, vmax=3.0)
    axB.set_xlim(0,feats_B.shape[0])
    axB.set_ylim(feats_B.shape[1], 0)
    axB.autoscale(False)

    axBstat = plt.subplot(gs[1,1], sharey=axB)
    axBstat.matshow(statify_feats(feats_B).T, vmin=-3.0, vmax=3.0)

    plt.savefig(_path(track, '_feats.png'), dpi=96, bbox_inches='tight')


def metric_strength(feats_A, feats_B, metric):
    # How well does this metric predict that paired clips from the same track
    # should be related to one another?
    Y = scipy.spatial.distance.cdist(feats_A, feats_B, metric)
    selfs = np.zeros(Y.shape[1], dtype=np.float)
    for i in range(len(selfs)):
        selfs[i] = Y[i, i]
    return (np.mean(Y) - np.mean(selfs)) / np.std(Y)


def rank_metrics(feats_A, feats_B):
    metrics = ['canberra', 'braycurtis', 'cityblock', 'euclidean',
        'correlation', 'cosine', 'sqeuclidean', 'seuclidean', 'chebyshev']
    scores = [(metric_strength(feats_A, feats_B, m), m) for m in metrics]
    return sorted(scores, key=lambda x:x[0], reverse=True)


def plot_comparisons(clips):
    fig = plt.figure(1, figsize=(1024/96,1024/96), dpi=96)
    plt.set_cmap('hot')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1,
        width_ratios=[1, 1], wspace=0.1)
    grids = [gs[0,0], gs[0,1], gs[1,0], gs[1,1]]

    all_A = np.array([statify_feats(f).ravel() for _,_,f,_,_ in clips])
    all_B = np.array([statify_feats(f).ravel() for _,_,_,_,f in clips])

    print "Comparison metric scores:"
    metric_scores = rank_metrics(all_A, all_B)
    for score, metric in metric_scores:
        print "  %s: %.3f" % (metric, score)

    for i, (score, metric) in enumerate(metric_scores[:4]):
        Y = scipy.spatial.distance.cdist(all_A, all_B, metric)
        ax = plt.subplot(grids[i])
        ax.set_aspect(1.)
        ax.matshow(Y)
        ax.axis('off')
        ax.text(1.0, -1.0, "%s: %.3f" % (metric, score))

    plt.savefig(_path("comparisons.png"), dpi=96, bbox_inches='tight')


def run(seed, n_tracks):
    tracks = list(library.tracks())
    random.seed(seed)
    random.shuffle(tracks)
    subset = tracks[:n_tracks]
    print "Reading tracks"
    clips = list()
    feats = list()
    for i, t in enumerate(subset):
        caption = _caption(t)
        print "  [%d/%d] %s" %(1+i, len(subset), caption)
        try:
            (clip_A, feats_A), (clip_B, feats_B) = get_clips(t)
            clips.append((t, clip_A, feats_A, clip_B, feats_B))
            feats.append(feats_A)
            feats.append(feats_B)
        except KeyboardInterrupt:
            sys.exit(0)

    allfeats = np.concatenate(feats, axis=0)
    # normalize: center each feature on its mean, scaled to its std
    allmeans = np.mean(allfeats, axis=0)
    allstds = np.std(allfeats, axis=0)
    allstats = np.vstack((allmeans, allstds))

    #plot_comparisons(clips)
    for i, (t, clip_A, feats_A, clip_B, feats_B) in enumerate(clips):
        norm_A = (feats_A - allmeans) / allstds
        norm_B = (feats_B - allmeans) / allstds
        clips[i] = (t, clip_A, norm_A, clip_B, norm_B)
    plot_comparisons(clips)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_tracks', type=int, default=32)
    kwargs = vars(parser.parse_args())
    run(**kwargs)


