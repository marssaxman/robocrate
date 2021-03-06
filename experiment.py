import os
import os.path
import sys
import library
import analysis
from musictoys import audiofile
import musictoys.analysis
import scipy.spatial
import numpy as np
import random
import argparse
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _cache(track, suffix):
    self_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(self_dir, "science-cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    name = track.caption + suffix
    return os.path.join(cache_dir, name)


def calc_clips(track, plot=False):
    # Load audio file and normalize for analysis
    signal = audiofile.read(track.source)
    signal = musictoys.analysis.normalize(signal)
    samplerate = signal.sample_rate
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
        fig = plt.figure(1, figsize=(1024/96, 1280/96), dpi=96)
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
        axStart.set_ylim([0, 1])
        axStart.get_xaxis().set_visible(False)
        axStart.axvline(best_A)
        axStart.text(best_A, 0, "A")
        axStart.axvline(best_B)
        axStart.text(best_B, 0, "B")

        axMatrix.matshow(sim_matrix)
        axMatrix.axis('off')

        plt.savefig(_cache(track, '.png'), dpi=96, bbox_inches='tight')

    return ((clip_A, feats_A), (clip_B, feats_B))


def read_tracks(subset):
    clips = list()
    todo = list()
    for t in subset:
        try:
            feats_A = np.load(_cache(t, "_A.npy"))
            feats_B = np.load(_cache(t, "_B.npy"))
            clip_A, sr_A = audiofile.read(_cache(t, "_A.wav"))
            clip_B, sr_A = audiofile.read(_cache(t, "_B.wav"))
            clips.append((t, clip_A, feats_A, clip_B, feats_B))
        except IOError:
            todo.append(t)
    if todo:
        print "Analyzing tracks, extracting representative clips:"
    for i, t in enumerate(todo):
        print "  [%d/%d] %s" % (1+i, len(todo), t.caption)
        (clip_A, feats_A), (clip_B, feats_B) = calc_clips(t)
        audiofile.write(_cache(t, "_A.wav"), clip_A, 22050)
        audiofile.write(_cache(t, "_B.wav"), clip_B, 22050)
        np.save(_cache(t, "_A.npy"), feats_A)
        np.save(_cache(t, "_B.npy"), feats_B)
        clips.append((t, clip_A, feats_A, clip_B, feats_B))
    return clips


def run(seed, n_tracks, experiment):
    tracks = list(library.tracks())
    random.seed(seed)
    random.shuffle(tracks)
    subset = tracks[:n_tracks]
    clips = read_tracks(subset)
    if experiment == 'similarity':
        from experiments.similarity import run
    elif experiment == 'clusters':
        from experiments.clusters import run
    elif experiment == 'features':
        from experiments.features import run
    elif experiment == 'dendrogram':
        from experiments.dendrogram import run
    elif experiment == 'tempogram':
        from experiments.tempogram import run
    run(clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_tracks', type=int, default=128)
    parser.add_argument('experiment',
        choices=['similarity', 'clusters', 'features', 'dendrogram',
                'tempogram'])
    kwargs = vars(parser.parse_args())
    try:
        run(**kwargs)
    except KeyboardInterrupt:
        sys.exit(0)

