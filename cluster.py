import os
import os.path
import sys
from musictoys import audiofile
import analysis
import numpy as np
import random
import sklearn.cluster
import sklearn.mixture
import sklearn.preprocessing
import sklearn.random_projection
import library
from collections import Counter

# info here:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html


def _calc_feats(path):
    data = audiofile.read(path)
    return analysis.extract(data, data.sample_rate)


def _caption(track):
    if track.title and track.artist:
        return "\"%s\" by %s" % (track.title, track.artist)
    if track.title:
        return track.title
    return os.path.splitext(os.path.basename(track.source))[0]


def _read_clips(tracks):
    # Read the audio summary for each track in the library.
    # Extract features. Return a list of feature summary vectors, with each
    # index corresponding to one item in the track list.
    feat_list = [None] * len(tracks)
    work_list = []
    for i, t in enumerate(tracks):
        # Try to load a saved npy array file containing the feature vector.
        path = t.summary
        featfile = os.path.splitext(path)[0] + '.npy'
        try:
            feat_list[i] = np.load(featfile)
        except IOError, ValueError:
            work_list.append((i, t))
    # Compute feature vectors for unprocessed files.
    for i, (feat_idx, t) in enumerate(work_list):
        print "[%d/%d] %s" % (i+1, len(work_list), _caption(t))
        try:
            featvec = _calc_feats(t.summary)
            featfile = os.path.splitext(t.summary)[0] + '.npy'
            np.save(featfile, featvec)
            feat_list[feat_idx] = featvec
        except KeyboardInterrupt:
            sys.exit(0)
    return feat_list


def _count(items):
    items = [i for i in items if not i is None]
    return len(items), Counter(items)


def _top3desc(items):
    total, counts = _count(items)
    top3 = counts.most_common(3)
    top3pct = [(k, v/float(total)) for k, v in top3]
    return ["%s (%.1f%%)" % (k, v*100.0) for k, v in top3pct]


def _feats_to_matrix(feat_list):
    featlin = list()
    for f in feat_list:
        if f.shape[1] < 59:
            f = np.pad(f, ((0,0), (0, 59-f.shape[1])), 'constant')
            assert (34,59) == f.shape
        featlin.append(f.ravel())
    # PCA expects (n_observations, n_dimensions) so we must put each track
    # in rows, so that feats.shape[0] == len(feat_list).
    feats = np.row_stack(featlin)
    return feats


def cluster():
    # Read the metadata describing the tracks in the library.
    tracks = library.tracks()
    # Read each audio summary clip and extract its feature series.
    feat_list = _read_clips(tracks)
    feats = _feats_to_matrix(feat_list)

    # Normalize the input data.
    feats = sklearn.preprocessing.scale(feats, axis=0)

    # Reduce dimensionality.
    transformer = sklearn.random_projection.SparseRandomProjection(eps=0.25)
    feats = transformer.fit_transform(feats)

    n_clusters = max(3, int(len(tracks) / 150.0))
    model = sklearn.mixture.GaussianMixture(
        n_components=n_clusters, covariance_type='full')

    print "fitting model"
    model.fit(feats)

    # Make predictions: which tracks should go into which clusters?
    print "generating predictions"
    labels = model.predict(feats)
    proba = model.predict_proba(feats)

    # Sort the tracks into groups, corresponding to each generated label.
    groups = [list() for i in range(n_clusters)]
    for i, track in enumerate(tracks):
        groups[labels[i]].append(track)

    # Print a report describing the crates we've found.
    for i, crate in enumerate(groups):
        if len(crate) < 4:
            continue
        print "Crate %d contains %d tracks" % (i, len(crate))

        # Print out a playlist with all these tracks.
        with open(os.path.join(os.getcwd(), "crate%d.m3u" % i), 'w') as fd:
            for t in crate:
                if not t.source is None:
                    fd.write(t.source.encode('utf-8') + '\n')

        # Which are the most-representative tracks in this cluster?
        clustermean = model.means_[i,:]
        #from sklearn.metrics.pairwise import euclidean_distances
        #clusterdist = euclidean_distances(feats, [clustermean])[:,0]
        from sklearn.metrics.pairwise import cosine_distances
        clusterdist = cosine_distances(feats, [clustermean])[:,0]

        bestfits = np.argsort(clusterdist)
        print "  Representative tracks:"
        for j in bestfits[:5]:
            info = tracks[j]
            print "    %s" % _caption(info)

        # Which are the most frequently represented artists?
        print "  Most frequent artists:"
        artists = Counter(t.artist for t in crate if not t.artist is None)
        print "    %s" % ", ".join(k for k,v in artists.most_common(12))
        print "  Most common genres:"
        print "    %s" % ", ".join(_top3desc(t.genre for t in crate))

        # What is a representative BPM range?
        if len(crate) > 1:
            bpms = [t.bpm for t in crate if not t.bpm is None]
            bpms = np.array(bpms, dtype=np.float)
            bpm_mean, bpm_std = np.mean(bpms), np.std(bpms)
            print "  Tempo: %d bpm (+/- %d)" % (bpm_mean, bpm_std)

        print ""
