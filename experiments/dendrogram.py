import os
import os.path
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def caption(track):
    if track.title and track.artist:
        return "%s - %s" % (track.artist, track.title)
    if track.title:
        return track.title
    return os.path.splitext(os.path.basename(track.source))[0]


def agglomerate(feats):
    return np.concatenate((feats.mean(axis=0), feats.std(axis=0)))


def run(clips):
    features = list()
    labels = list()
    for t, clip_A, feats_A, clip_B, feats_B in clips:
        features.append(agglomerate(feats_A))
        labels.append(caption(t))
    features = np.array(features)
    distance = scipy.spatial.distance.pdist(features, 'cosine')

    plt.clf()
    plt.figure(1, figsize=(48,24))
    Z = linkage(distance, 'ward', optimal_ordering=True)
    with plt.rc_context({'lines.linewidth': 0.5}):
        dn = dendrogram(Z, orientation='left', labels=labels)

    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=2)

    plt.tight_layout()
    plt.savefig("dendrogram.svg")

