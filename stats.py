import numpy as np
import library
import os.path
import features
import argparse
import scipy.stats

# Exploratory statistics: what's in that library we collected?


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def ns(n):
    return (("%.3f" if 0.01 <= np.abs(n) < 100 else "%.3e") if n else "%d") % n


def print_feat(i, feats):
    name = features.names()[i]
    avg = feats[:,i].mean()
    dev = feats[:,i].std()
    scale = (dev / np.abs(avg)) * 100.0
    print("    %s (%s . %s, %.2f%%)" % (name, ns(avg), ns(dev), scale))


def deviation_report(feats):
    # print the outliers in terms of standard deviation magnitude
    deviation = feats.std(axis=0)
    average = feats.mean(axis=0)
    indexes = np.arange(feats.shape[-1])
    # remove values where either the mean or the deviation are zero
    usable = (deviation != 0) & (average != 0)
    deviation = np.compress(usable, deviation)
    average = np.compress(usable, average)
    indexes = np.compress(usable, indexes)

    num = 10

    ordering = indexes[np.argsort(deviation)]
    print("top %d most deviant features, absolute scale" % num)
    for i in ordering[::-1][:num]:
        print_feat(i, feats)
    print("bottom %d least deviant features, absolute scale" % num)
    for i in ordering[:num]:
        print_feat(i, feats)

    scaled_deviation = np.divide(deviation, np.abs(average))
    ordering = indexes[np.argsort(scaled_deviation)]
    print("top %d most deviant features, relative scale" % num)
    for i in ordering[::-1][:num]:
        print_feat(i, feats)
    print("bottom %d least deviant features, relative scale" % num)
    for i in ordering[:num]:
        print_feat(i, feats)


def mean_stdev_limits_report(feats, *args, **kwargs):
    print("mean, stdev, and limits for each feature")
    names = features.names()
    for i in np.arange(feats.shape[-1]):
        feat = feats[:,i]
        minv, maxv = feat.min(), feat.max()
        meanv, stdv = feat.mean(), feat.std()
        print("%s: (%s .. %s); mean=%s, stdev=%s " % (
            names[i], ns(minv), ns(maxv), ns(meanv), ns(stdv)))


def extreme_distributions(feats):
    # which average values are the most extreme? we want the ones closest to
    # zero and the ones closest to 1
    actual_avg = feats.mean(axis=0)
    new_feats = np.subtract(feats, feats.min(axis=0))
    new_feats = np.divide(new_feats, new_feats.max(axis=0))
    avgoutlier = new_feats.mean(axis=0)
    highavg = avgoutlier > 0.5
    avgoutlier[highavg] = 1.0 - avgoutlier[highavg]
    ordering = np.argsort(avgoutlier)
    ordering = np.compress(actual_avg[ordering] != 0, ordering)
    print("top 20 most extreme distributions")
    for i in ordering[:20]:
        print_feat(i, feats)


def correlation_report(feats, num=20):
    R = np.corrcoef(feats, rowvar=False)

    fig = plt.figure(1, figsize=(1280/64, 1280/64), dpi=96)
    plt.matshow(R)
    plt.gca().set_aspect(1.)
    plt.gca().axis('off')
    plt.savefig("correlation.png", dpi=96, bbox_inches='tight')

    # we only need half of this matrix, because it is symmetrical
    R = np.triu(R, k=1)
    # we only care about magnitude of correlation, not direction
    flatR = R.ravel()
    np.absolute(flatR, out=flatR, where=np.isfinite(flatR))
    ordering = np.argsort(flatR)
    ordering = np.compress(np.isfinite(flatR[ordering]), ordering)
    names = features.names()

    print("top %d most highly correlated variables" % num)
    for flat in ordering[::-1][:num]:
        pair = np.unravel_index(flat, R.shape)
        coeff = R[pair]
        print("    %s . %s: %s" % (names[pair[0]], names[pair[1]], ns(coeff)))
    print("bottom %d least highly correlated variables" % num)
    for flat in ordering[:num]:
        pair = np.unravel_index(flat, R.shape)
        coeff = R[pair]
        print("    %s . %s: %s" % (names[pair[0]], names[pair[1]], ns(coeff)))


def kurtosis_report(feats, num=20):
    # which are the most and the least gaussian features present?
    mean = feats.mean(axis=0)
    var = feats.var(axis=0)
    diffmean = feats - mean
    indexes = np.arange(feats.shape[-1])
    usable = (mean != 0) & (var != 0)
    mean = np.compress(usable, mean)
    var = np.compress(usable, var)
    diffmean = np.compress(usable, diffmean)
    indexes = np.compress(usable, indexes)

    kurt = (1. / feats.shape[-1]) * np.sum(diffmean ** 4) / (var ** 2) - 3.0
    ordering = np.argsort(kurt)
    print("top %d most gaussian features" % num)
    names = features.names()
    for i in ordering[::-1][:num]:
        print("    %s (%s)" % (names[indexes[i]], ns(kurt[i])))
    print("bottom %d least gaussian features" % num)
    for i in ordering[:num]:
        print("    %s (%s)" % (names[indexes[i]], ns(kurt[i])))



def normaltest_report(feats, num=20):
    # to what degree does each feature represent a normal distribution?
    numfeats = feats.shape[-1]
    statistic = np.zeros(numfeats)
    pvalue = np.zeros(numfeats)
    for i in np.arange(numfeats):
        s, p = scipy.stats.normaltest(feats[:,i])
        statistic[i] = s
        pvalue[i] = p
        print("    %s s=%s, p=%s" % (features.names()[i], ns(s), ns(p)))


def tags_report(feats, num=15):
    # compute a score for each tag and pick the ones that meet some threshold.
    tags = library.tags()
    meansquare = sum(len(v) ** 2 for v in tags.itervalues()) / len(tags)
    significance = int(meansquare ** 0.5)
    tags = [(k, v) for k, v in tags.iteritems() if len(v) > significance]
    # compute the mean and standard deviation for each feature.
    # for each tag, compute the mean for each track associated with that tag.
    # select features whose tag mean is more distant from the library mean
    # than the standard deviation.
    lib_mean = feats.mean(axis=0)
    lib_std = feats.std(axis=0)
    threshold = lib_std * 1.5
    # get the index for each track
    track_map = dict()
    for i, t in enumerate(library.tracks()):
        track_map[t.hash] = i
    # for each tag, make a mask with the indexes of its tracks
    names = features.names()
    for tag, vals in tags:
        print("tag %s is associated with %d tracks" % (tag, len(vals)))
        indexes = np.array([track_map[t.hash] for t in vals])
        tag_mean = feats[indexes,:].mean(axis=0)
        outliers = np.argwhere(np.absolute(tag_mean - lib_mean) > threshold)
        for i in outliers[...,0]:
            print("    %s local mean=%.2f; library mean=%.2f" % (
                names[i], tag_mean[i], lib_mean[i]))


def run(report, **kwargs):
    feats = features.matrix(library.tracks())
    report(feats, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    report_list = {
        'correlation': correlation_report,
        'deviation': deviation_report,
        'kurtosis': kurtosis_report,
        'normaltest': normaltest_report,
        'tags': tags_report,
        'mean_stdev_limits': mean_stdev_limits_report,
    }
    parser.add_argument('report', choices=report_list)
    parser.add_argument('--num', type=int, default=10)

    args = vars(parser.parse_args())
    report = report_list[args.pop('report')]
    run(report, **args)

