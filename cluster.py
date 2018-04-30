import os
import os.path
import scipy.io.wavfile
import analysis
import config
import numpy as np
import random
import sklearn.cluster
import json


def _read_clips():
    worklist = list()
    for name in os.listdir(config.dir):
        if not name.endswith(".wav"):
            continue
        worklist.append(name)
    random.shuffle(worklist)
    featvec = list()
    hashes = list()
    for i, name in enumerate(worklist):
        print "[%d/%d] %s" % (i+1, len(worklist), name)
        # Trim off the ".wav" type and the "_L" or "_R" suffix if present.
        hash = name[:-4]
        if hash[-2] == '_':
            hash = hash[:-2]
        # Go load the wav file and extract its audio features.
        path = os.path.join(config.dir, name)
        samplerate, data = scipy.io.wavfile.read(path)
        if len(data) < 1024:
            print "file too short: %s is only %d bytes" % (name, len(data))
            continue
        feats = analysis.extract(data, samplerate)
        # we generate 30-second summaries using 34 features, so our target shape
        # is always 34x59. We shouldn't be generating any smaller samples, but
        # there is currently a bug in the clip generator. On the other hand,
        # we're just going to average it all out anyway, so it doesn't matter.
        norms = np.mean(feats, axis=1)
        featvec.append(norms)
        hashes.append(hash)
    return np.array(featvec), hashes


def _get_id3_genre(hash):
    path = os.path.join(config.dir, hash + "_ID3.json")
    with open(path, 'r') as fd:
        tags = json.load(fd)
        return tags.get('genre', None)


def cluster():
    feats, hashes = _read_clips()
    print "  loaded clips: %s" % str(feats.shape)
    n_clusters = 8
    model = sklearn.cluster.KMeans(n_clusters=n_clusters)
    print "  fitting model"
    model.fit(feats)
    labels = model.predict(feats)
    # Build a list of hashes for each category.
    groups = [list() for i in range(n_clusters)]
    for i, hash in enumerate(hashes):
        groups[labels[i]].append(hash)
    # List the number of items in each category.
    for i, group in enumerate(groups):
        print "Cluster %d contains %d items" % (i, len(group))
        clustertags = dict()
        for hash in group:
            tag = _get_id3_genre(hash)
            if not tag:
                continue
            clustertags[tag] = 1 + clustertags.get(tag, 0)
        tagcounts = sorted(clustertags.items(), key=lambda x:x[1], reverse=True)
        toptags = tagcounts[:min(len(tagcounts), 3)-1]
        summary = ["%s (%d)" % (g, n) for (g, n) in toptags]
        print "  genres: %s" % ", ".join(summary)

