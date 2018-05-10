import scipy.spatial
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def statify_feats(feats):
    # There is no measurable difference between stacking the feature values
    # horizontally or vertically.
    return np.vstack((
        np.mean(feats, axis=0),
        np.std(feats, axis=0)
    ))


def metric_strength(feats_A, feats_B, metric):
    # We have compared pairs of clips from an array of tracks. Columns and rows
    # share the same order. We expect that the distance between clips from the
    # same track will be substantially smaller than distances to other tracks.
    # How well does this metric predict that paired clips from the same track
    # should be related to one another?
    Y = scipy.spatial.distance.cdist(feats_A, feats_B, metric)
    selfs = np.zeros(Y.shape[1], dtype=np.float)
    for i in range(len(selfs)):
        selfs[i] = Y[i, i]
    return (np.mean(Y) - np.mean(selfs)) / np.std(Y)


def rank_metrics(feats_A, feats_B):
    metrics = ['canberra', 'braycurtis', 'cityblock', 'euclidean', 'chebyshev',
               'correlation', 'cosine', 'sqeuclidean', 'seuclidean']
    scores = [(metric_strength(feats_A, feats_B, m), m) for m in metrics]
    return sorted(scores, key=lambda x: x[0], reverse=True)


def run(clips):
    # normalize: center each feature on its mean, scaled to its std
    feats = list()
    for t, clip_A, feats_A, clip_B, feats_B in clips:
        feats.append(feats_A)
        feats.append(feats_B)
    allfeats = np.concatenate(feats, axis=0)
    allmeans = np.mean(allfeats, axis=0)
    allstds = np.std(allfeats, axis=0)
    for i, (t, clip_A, feats_A, clip_B, feats_B) in enumerate(clips):
        norm_A = (feats_A - allmeans) / allstds
        norm_B = (feats_B - allmeans) / allstds
        clips[i] = (t, clip_A, norm_A, clip_B, norm_B)

    # try out each similarity algorithm and rank them, scoring them according
    # to their ability to match the corresponding A and B clips for each track
    fig = plt.figure(1, figsize=(1024/96, 1024/96), dpi=96)
    plt.set_cmap('hot')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1,
                           width_ratios=[1, 1], wspace=0.1)
    grids = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]

    all_A = np.array([statify_feats(f).ravel() for _, _, f, _, _ in clips])
    all_B = np.array([statify_feats(f).ravel() for _, _, _, _, f in clips])

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

    plt.savefig("similarity.png", dpi=96, bbox_inches='tight')

