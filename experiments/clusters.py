import numpy as np
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering
from time import time

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


def run(clips):
    X = list()
    for t, clip_A, feats_A, clip_B, feats_B in clips:
        X.append(statify_feats(feats_A).ravel())
    X = np.array(X)
    print "X.shape == %s" % str(X.shape)

    print("Computing embedding")
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    print("Done.")

    for linkage in ('ward', 'average', 'complete'):
        plt.clf()
        fig = plt.figure(1, figsize=(1024/96, 1024/96), dpi=96)
        plt.set_cmap('hot')

        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        t0 = time()
        clustering.fit(X_red)
        print("%s : %.2fs" % (linkage, time() - t0))

        labels = clustering.labels_
        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
        X_red = (X_red - x_min) / (x_max - x_min)

        for i in range(X_red.shape[0]):
            plt.text(X_red[i, 0], X_red[i, 1], '+',
                    color=plt.cm.spectral(labels[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

        plt.axis('off')
        plt.tight_layout()
        plt.savefig("clusters-%s.png" % linkage, dpi=96)

