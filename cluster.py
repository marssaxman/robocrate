import os
import os.path
import scipy.io.wavfile
import analysis
import config
import numpy as np
import random
import sklearn.cluster
import sklearn.preprocessing
import json
import library
from collections import Counter


def log(msg):
    if config.verbose:
        print msg


def _calc_feats(path):
    samplerate, data = scipy.io.wavfile.read(path)
    featseries = analysis.extract(data, samplerate)
    # We'll take the average of each feature to represent the track.
    # Might be worth doubling the feature array, including variance too.
    featvec = np.mean(featseries, axis=1)
    return featvec


def _read_clips(tracks):
    # Read the audio summary for each track in the library.
    # Extract features. Return a list of feature summary vectors, with each
    # index corresponding to one item in the track list.
    feat_list = [None] * len(tracks)
    for i, t in enumerate(tracks):
        print "[%d/%d] %s" % (i+1, len(tracks), t.caption)
        feat_list[i] = _calc_feats(t.summary)
    return feat_list


def _count(items):
    items = [i for i in items if not i is None]
    return len(items), Counter(items)


def _top3desc(items):
    total, counts = _count(items)
    top3 = counts.most_common(3)
    top3pct = [(k, v/float(total)) for k, v in top3]
    return ["%s (%.1f%%)" % (k, v*100.0) for k, v in top3pct]


def cluster():
    # Read the metadata describing the tracks in the library.
    tracks = library.tracks()
    # Read each audio summary clip and extract its feature series.
    feat_list = _read_clips(tracks)

    # Build a classifier. We'll eventually do something clever like using the
    # elbow method, but for now we'll assume there should be roughly 200 tracks
    # in each cluster.
    n_clusters = max(3, int(len(tracks) / 200.0))
    model = sklearn.cluster.KMeans(n_clusters=n_clusters)
    # Convert the list of feature vectors into a matrix and fit the model.
    # Perhaps we should normalize? PCA might also be valuable.
    log("fitting model")
    feats = np.array(feat_list)
    feats = sklearn.preprocessing.scale(feats, axis=0)
    model.fit(feats)

    # Make predictions: which tracks should go into which clusters?
    log("generating predictions")
    labels = model.predict(feats)
    # Sort the tracks into groups, corresponding to each generated label.
    groups = [list() for i in range(n_clusters)]
    for i, track in enumerate(tracks):
        groups[labels[i]].append(track)

    # Print a report describing the crates we've found.
    for i, crate in enumerate(groups):
        print "Crate %d contains %d tracks" % (i, len(crate))
        # Which are the most-representative tracks in this cluster?
        distances = model.transform(feats)[:, i]
        bestfits = np.argsort(distances)[::][:5]
        print "  Representative tracks:"
        for j in bestfits:
            info = tracks[j]
            print "    %s" % info.caption

        # Which are the most frequently represented artists?
        print "  Most frequent artists:"
        artists = Counter(t.artist for t in crate if not t.artist is None)
        print "    %s" % ", ".join(k for k,v in artists.most_common(3))
        print "  Most common genres:"
        print "    %s" % ", ".join(_top3desc(t.genre for t in crate))

        # What is a representative BPM range?
        bpms = [t.bpm for t in crate if not t.bpm is None]
        bpms = np.array(bpms, dtype=np.float)
        bpm_mean, bpm_std = np.mean(bpms), np.std(bpms)
        bpm_lo = int(round(bpm_mean - bpm_std))
        bpm_hi = int(round(bpm_mean + bpm_std))
        print "  Tempo range: %d-%d" % (bpm_lo, bpm_hi)

        print ""
