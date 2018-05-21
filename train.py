import numpy as np
import library
import os.path
import features
import argparse
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import MultiTaskLassoCV
from collections import namedtuple
import scipy.stats

Dataset = namedtuple('Dataset', 'input target')


# We'll use artist names, publishers, and genre names as label classes, then
# train models and evaluate features to select a classifier for this dataset.

def collect_labels(libtracks, num_labels=None):
    # Make an array of labels for training. Use the artist and genre tags
    # from the ID3 metadata.
    labels = library.tags()

    if num_labels is None:
        # We expect to have a long-tail distribution of tag population. Take the
        # RMS average of the label population and use that as a threshold; this
        # will leave us with a good handful of well-populated categories.
        meansquare = sum(
            len(v) ** 2 for v in labels.itervalues()) / len(labels)
        threshold = int(meansquare ** 0.5)
        labels = [(k, v) for k, v in labels.iteritems() if len(v) > threshold]
        print("Selecting the %d most common tags, with at least %d tracks each):" %
              (len(labels), threshold))
    else:
        # The user has requested a specific number of category labels. Sort the
        # list, throw away the outliers on either end, and return some from
        # the more-commonly-represented end of the normal range.
        labels = sorted(labels.items(), key=lambda x: len(x[1]), reverse=True)
        if num_labels * 2 < len(labels):
            # throw away the leading outliers
            del labels[:num_labels]
        del labels[num_labels:]
        print("Selecting %d representative tags:" % len(labels))

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


def split_dataset(data, *args, **kwargs):
    in_train, in_test, target_train, target_test = \
        train_test_split(data.input, data.target, *args, **kwargs)
    return Dataset(in_train, target_train), Dataset(in_test, target_test)


def transform_input(processor, data):
    return Dataset(processor.transform(data.input), data.target)


def searchreport(results, top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def gridsearch(model, data, param_grid):
    gscv = GridSearchCV(model, param_grid=param_grid, verbose=1)
    gscv.fit(*data)
    searchreport(gscv.cv_results_)


def randomsearch(model, data, n_iter, param_dist):
    rscv = RandomizedSearchCV(model, param_distributions=param_dist,
                              n_iter=n_iter, verbose=1)
    rscv.fit(*data)
    print rscv.cv_results_


def train(num_labels=None, gridcv=False, randomcv=False):
    # Load the track library. Collect metadata labels. Generate a target
    # matrix. Load features for each track in the target matrix.
    libtracks = library.tracks()
    labels = collect_labels(libtracks, num_labels)
    tracklist, target = generate_target(labels)
    data = Dataset(features.matrix(tracklist), target)
    feat_names = features.names()
    train, test = split_dataset(data, test_size=0.4)  # , random_state=0)
    # A random forest should be able to handle the excessive dimensionality
    # of our dataset relative to the number of samples.
    clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, verbose=1)

    if randomcv:
        print "random parameter search..."
        randomsearch(clf, train, 20, {
            "max_depth": [3, None],
            "max_features": scipy.stats.randint(50, 100),
            "min_samples_split": scipy.stats.randint(2, 11),
            "min_samples_leaf": scipy.stats.randint(1, 11),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        })

    if gridcv:
        print "grid parameter search..."
        gridsearch(clf, train, {
            "max_depth": [3, None],
            "max_features": [50, 75, 100],
            "min_samples_split": [2, 3, 10],
            "min_samples_leaf": [1, 3, 10],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        })

    print("training classifier...")
    clf.fit(*train)
    mean_importance = clf.feature_importances_.mean()
    # Measure prediction accuracy for the original training run.
    pred_target = clf.predict(test.input)
    orig_score = accuracy_score(test.target, pred_target)
    print("accuracy score with %d features: %.2f%%" %
          (len(feat_names), orig_score * 100.0))

    # Reduce the feature set.
    print("selecting best features...")
    sfm = SelectFromModel(clf, threshold='1.5*mean')
    sfm.fit(*train)
    # Print the names of the most important features
    feature_subset = sfm.get_support(indices=True)
    for i in feature_subset:
        importance = clf.feature_importances_[i] / mean_importance
        print "    %.1f: '%s'" % (importance, feat_names[i])

    # make a new training set with just the useful features.
    print("preparing new training subset...")
    slim_train = transform_input(sfm, train)
    slim_test = transform_input(sfm, test)
    feat_names = [feat_names[i] for i in feature_subset]

    # train a new classifier using the reduced feature set.
    print("training subset classifier...")
    clf_slim = RandomForestClassifier(n_estimators=120, n_jobs=-1, verbose=1)
    clf_slim.fit(*slim_train)

    # measure accuracy of the retrained models
    pred_slim = clf_slim.predict(slim_test.input)
    slim_score = accuracy_score(slim_test.target, pred_slim)
    print("subset accuracy with %d features: %.2f%%" %
          (len(feature_subset), slim_score * 100.0))


# A variety of learning experiments follow

def reduce_kbest(data, feat_names):
    print("reducing features with SelectKBest/chi2")
    skb = SelectKBest(chi2, k=200)
    scaled = Dataset(preprocessing.minmax_scale(data.input), data.target)
    skb.fit(*scaled)
    print("feature reduction complete")
    data = transform_input(skb, data)
    subset = skb.get_support(indices=True)
    print("skb.scores_.shape", skb.scores_.shape)
    print("subset.shape", subset.shape)
    for i in subset:
        print("    %.1f: '%s'" % (skb.scores_[i], feat_names[i]))
    return data, [feat_names[i] for i in subset]


def add_arguments(parser):
    parser.add_argument('--num_labels', type=int, default=None)
    parser.add_argument('--gridcv', default=False, action='store_true')
    parser.add_argument('--randomcv', default=False, action='store_true')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    train(**var(parser.parse_args()))
