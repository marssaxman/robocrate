import numpy as np
import library
import os.path
import features
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import MultiTaskLassoCV

# Use a random forest classifier to evaluate our massive array of features.
# We'll use artist names and genre tags as training labels.


def collect_labels(libtracks):
    # Make an array of labels for training. Use the artist and genre tags
    # from the ID3 metadata.
    labels = dict()
    for t in libtracks:
        names = set()
        # multiple artists are often packed in for a single track
        if t.artist:
            names.update(t.artist.split(", "))
        if t.album_artist:
            names.update(t.album_artist.split(", "))
        if t.remixer:
            names.update(t.remixer.split(", "))
        if t.genre:
            names.add(t.genre)
        if t.publisher:
            names.add(t.publisher)
        for n in names:
            if n in labels:
                labels[n].append(t)
            else:
                labels[n] = [t]
    # We expect to have a long-tail distribution of tag population. Take the
    # RMS average of the label population and use that as a threshold; this
    # will leave us with a good handful of well-populated categories.
    meansquare = sum(len(v) ** 2 for v in labels.itervalues()) / len(labels)
    threshold = int(meansquare ** 0.5)
    labels = [(k, v) for k, v in labels.iteritems() if len(v) > threshold]
    print("Selecting the %d most common tags (at least %d tracks each):" % \
        (len(labels), threshold))
    for k, v in labels:
        print("    '%s': %d" % (k, len(v)))

    # Count the proportion of tracks we'll be using
    keephashes = set()
    for _, v in labels:
        keephashes.update(t.hash for t in v)
    numkeep = len(keephashes)
    print("Using %d tracks out of %d (%.1f%% of the library)" % (
        numkeep, len(libtracks), (numkeep * 100.0 / len(libtracks))))
    return labels


def generate_target(labels):
    # Given a list of labels, with a list of tracks each one contains,
    # generate a target data array. This should be a one-hot array so we can
    # do multiclass learning, but I suppose I'll figure that out later.
    # Start by reversing the index, from tracks to labels.
    tracklist = dict()
    for i, (name, tracks) in enumerate(labels):
        for t in tracks:
            if t.hash in tracklist:
                _, tracklabels = tracklist[t.hash]
                tracklabels.append(name)
            else:
                tracklist[t.hash] = (t, [name])
    # Generate a one-hot class array matching tracks with labels.
    mlb = MultiLabelBinarizer()
    target = mlb.fit_transform(labels for _, labels in tracklist.itervalues())
    # Now we have an ordered list of tracks which appear at least once in the
    # labels, and we have a corresponding target array with category data.
    return (t for t, _ in tracklist.itervalues()), target


def reduce_kbest(feats, target, feat_names):
    print("reducing features with SelectKBest/chi2")
    skb = SelectKBest(chi2, k=200)
    scaled_feats = preprocessing.minmax_scale(feats)
    skb.fit(scaled_feats, target)
    print("feature reduction complete")
    feats = skb.transform(feats)
    subset = skb.get_support(indices=True)
    print("skb.scores_.shape", skb.scores_.shape)
    print("subset.shape", subset.shape)
    for i in subset:
        print("    %.1f: '%s'" % (skb.scores_[i], feat_names[i]))
    return feats, [feat_names[i] for i in subset]


def train():
    libtracks = library.tracks()
    labels = collect_labels(libtracks)
    tracklist, target = generate_target(labels)
    feats = features.matrix(tracklist)
    feat_names = features.names()

    #feats, feat_names = reduce_kbest(feats, target, feat_names)

    # We have labels and a data set. Split into test & training sets.
    feats_train, feats_test, target_train, target_test = \
        train_test_split(feats, target, test_size=0.4, random_state=0)

    print("training classifier...")
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    clf.fit(feats_train, target_train)
    mean_importance = clf.feature_importances_.mean()
    # Measure prediction accuracy for the original training run.
    target_pred = clf.predict(feats_test)
    orig_score = accuracy_score(target_test, target_pred)
    print("accuracy score with %d features: %.2f%%" % \
        (len(feat_names), orig_score * 100.0))

    # Reduce the feature set.
    print("selecting best features (threshold=%.2e)..." % mean_importance)
    sfm = SelectFromModel(clf, threshold=mean_importance)
    sfm.fit(feats_train, target_train)
    # Print the names of the most important features
    feature_subset = sfm.get_support(indices=True)
    # for i in feature_subset:
    #    importance = clf.feature_importances_[i] / mean_importance
    #    print "    %.1f: '%s'" % (importance, feat_names[i])

    # make a new training set with just those features
    print("preparing new training subset...")
    slim_feats_train = sfm.transform(feats_train)
    slim_feats_test = sfm.transform(feats_test)

    # train a new classifier using the reduced feature set
    print("training subset classifier...")
    clf_slim = RandomForestClassifier(
        n_estimators=10000, random_state=0, n_jobs=-1)
    clf_slim.fit(slim_feats_train, target_train)

    # measure accuracy of the retrained models
    slim_pred = clf_slim.predict(slim_feats_test)
    slim_score = accuracy_score(target_test, slim_pred)
    print("subset accuracy with %d features: %.2f%%" % \
        (len(feature_subset), slim_score * 100.0))


if __name__ == '__main__':
    train()
