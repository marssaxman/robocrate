import os
import os.path
import numpy as np
from analysis import mfcc, spectral
from scipy.fftpack import dct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# interesting research on feature weighting:
# http://journals.sagepub.com/doi/pdf/10.1177/1029864916655596

# some strategies for feature selection:
# https://www.researchgate.net/publication/308483875_A_Study_on_Feature_Selection_and_Classification_Techniques_of_Indian_Music


def _caption(track):
    if track.title and track.artist:
        return "%s - %s" % (track.artist, track.title)
    if track.title:
        return track.title
    return os.path.splitext(os.path.basename(track.source))[0]


def statify_feats(feats):
    # There is no measurable difference between stacking the feature values
    # horizontally or vertically.
    return np.vstack((
        np.mean(feats, axis=0),
        np.std(feats, axis=0)
    ))


def plot_features(track, feats_A, feats_B):
    fig = plt.figure(1, figsize=(1024/96, 1280/96), dpi=96)
    plt.set_cmap('hot')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1,
                           width_ratios=[12, 1], wspace=0.1)
    axA = plt.subplot(gs[0, 0])
    axA.matshow(np.clip(feats_A.T, -3.0, 3.0), vmin=-3.0, vmax=3.0)
    axA.set_xlim(0, feats_A.shape[0])
    axA.set_ylim(feats_A.shape[1], 0)
    axA.autoscale(False)

    axAstat = plt.subplot(gs[0, 1], sharey=axA)
    axAstat.matshow(statify_feats(feats_A).T, vmin=-3.0, vmax=3.0)

    axB = plt.subplot(gs[1, 0], sharex=axA)
    axB.matshow(np.clip(feats_B.T, -3.0, 3.0), vmin=-3.0, vmax=3.0)
    axB.set_xlim(0, feats_B.shape[0])
    axB.set_ylim(feats_B.shape[1], 0)
    axB.autoscale(False)

    axBstat = plt.subplot(gs[1, 1], sharey=axB)
    axBstat.matshow(statify_feats(feats_B).T, vmin=-3.0, vmax=3.0)

    plt.savefig(_caption(track)+'_feats.png', dpi=96, bbox_inches='tight')


def hamming(N):
    # improved hamming window: original implementation used 0.54, 0.46
    i = np.arange(N).astype(np.float)
    return 0.53836 - (0.46164 * np.cos(np.pi * 2.0 * i / (N-1)))


def altfeats(clip):

    def zcr(frame):
        count = len(frame)
        count_zeros = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        return (np.float64(count_zeros) / np.float64(count-1.0))

    def energy(frame):
        return np.sum(frame ** 2) / np.float64(len(frame))

    frame_len = 2048
    n_fft = 1024
    window = hamming(2048)
    [fbank, freqs] = mfcc.init(22050.0, n_fft)
    clipfeats = list()
    s_prev = None
    for i in xrange(0, len(clip)-frame_len+1, frame_len):
        frame = clip[i:i+frame_len] * window
        s = np.abs(np.fft.rfft(frame))[:n_fft] / float(n_fft)
        [centroid, spread] = spectral.centroid_and_spread(s, 22050)
        entropy = spectral.entropy(s)
        flux = spectral.flux(s, s_prev) if not s_prev is None else 0.0
        rolloff = spectral.rolloff(s, 0.90)
        mfccs = mfcc.filter(s, fbank, 13)
        spectrals = [centroid, spread, entropy, flux, rolloff]
        framefeats = np.concatenate(([zcr(s), energy(s)], spectrals, mfccs))
        clipfeats.append(framefeats)
        s_prev = s
    return np.array(clipfeats)


def stft(clip, frame_len=1024):
    # generator for magnitude spectrum frames from an audio clip
    # assume clip is 22050Hz mono; normalize it
    clip = clip.astype(np.float)
    clip -= clip.mean()
    clip /= np.abs(clip).max()
    # assume half-overlap
    step_len = int(frame_len / 2)
    window = hamming(frame_len)
    num_frames = int(np.floor((len(clip) - frame_len + 1) / step_len))
    for i in xrange(num_frames):
        offset = i * step_len
        # get the samples and multiply by the window to reduce ringing
        frame = clip[offset:offset+frame_len] * window
        # get the magnitude spectrum
        s = np.abs(np.fft.rfft(frame))
        yield s


def hertz_to_mel(freq):
    return 2595.0 * np.log10(1 + (freq/700.0))


def mel_to_hertz(mel):
    return 700.0 * (10**(mel/2595.0)) - 700.0


def melfilter(num_mels, num_ffts, samplerate, freq_min, freq_max):
    # compute the mel values for the specified frequency range
    mel_max = hertz_to_mel(freq_max)
    mel_min = hertz_to_mel(freq_min)
    delta_mel = np.abs(mel_max - mel_min) / (num_mels + 1.0)
    # generate triangular filter center, lower, and upper freqs for each band
    freqs = mel_to_hertz(mel_min + delta_mel * np.arange(0, num_mels+2))
    edges = zip(freqs[1:-1], freqs[:-2], freqs[2:])
    # populate a filterbank matrix
    fbank = np.zeros((num_mels, num_ffts))
    fft_freqs = np.linspace(0.0, samplerate/2.0, num_ffts)
    for mel_band, (center, lower, upper) in enumerate(edges):
        # the rising slope includes all bins from the lower edge to the center
        rising = (fft_freqs >= lower) == (fft_freqs <= center)
        fbank[mel_band, rising] = (
            (fft_freqs[rising] - lower) / (center - lower)
        )
        # the falling slope includes FFT bins from the center to the upper edge
        falling = (fft_freqs >= center) == (fft_freqs <= upper)
        fbank[mel_band, falling] = (
            (upper - fft_freqs[falling]) / (upper - center)
        )
    return fbank


mel_fbank = melfilter(24, 513, 22050.0, 64.0, 11025.0)


def mfcc(clip):
    # assume clip is 22050Hz mono; normalize it
    clip = clip.astype(np.float)
    clip -= clip.mean()
    clip /= np.abs(clip).max()
    # we'll take 1024-sample FFTs every 512 samples
    frame_len = 1024
    step_len = 512
    window = hamming(frame_len)
    num_frames = int(np.floor((len(clip) - frame_len + 1) / step_len))
    num_ceps = 13
    clip_ceps = np.zeros((num_frames, num_ceps))
    for i in xrange(num_frames):
        offset = i * step_len
        # get the samples and multiply by the window to reduce ringing
        frame = clip[offset:offset+frame_len] * window
        # get the magnitude spectrum
        s = np.abs(np.fft.rfft(frame))
        # compute power
        s = np.square(s) / float(frame_len)
        # apply mel-scaled filter to get mel-band energies for this frame
        mels = np.dot(s, mel_fbank.T)
        # take log to account for perceptual nonlinearity
        mels = np.log(mels + np.finfo(np.float).eps)
        # take DCT to convert into cepstrum domain
        frame_ceps = dct(mels, type=2, norm='ortho')
        # should we 'lifter' the ceps here?
        # in any case, skip c0 (total energy) and the higher-order ceps
        clip_ceps[i,:] = frame_ceps[1:1+num_ceps]
    return clip_ceps


def mfccfeats(clip):
    # Compute the MFCC series for this clip.
    # For each coefficient, return the mean and standard deviation, plus the
    # mean difference between adjacent coefficients.
    ceps = mfcc(clip)
    diffs = np.abs(ceps[:-1,:] - ceps[1:,:])
    feats = (ceps.mean(axis=0), ceps.std(axis=0), diffs.mean(axis=0))
    return np.concatenate(feats)


def tempogram(novelty):
    # compute a tempogram over the novelty function, using 256 steps of 512
    # samples each, which is roughly six seconds; enough for tempo information
    # and a sense of the general rhythm pattern.
    step_len = 256
    num_frames = int(np.floor((len(novelty) - step_len + 1) / step_len))
    tempogram = np.zeros(step_len, dtype=np.float)
    for i in xrange(num_frames):
        offset = i * step_len
        frame = novelty[offset:offset+step_len]
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[corr.size / 2:]
        tempogram += corr / np.max(np.abs(corr))
    # Since we've been summing each tempogram as we go, we can get the average
    # now by dividing over the number of frames.
    return tempogram / float(num_frames)


def rhythmfeats(clip):
    # Compute an activation tempogram for this clip.
    # assume clip is 22050Hz mono; normalize it
    clip = clip.astype(np.float)
    clip -= clip.mean()
    clip /= np.abs(clip).max()
    # we'll take 1024-sample FFTs every 512 samples
    frame_len = 1024
    step_len = 512
    window = hamming(frame_len)
    num_frames = int(np.floor((len(clip) - frame_len + 1) / step_len))
    prev = None
    novelty = np.zeros(num_frames, dtype=np.float)
    for i in xrange(num_frames):
        offset = i * step_len
        # get the samples and multiply by the window to reduce ringing
        frame = clip[offset:offset+frame_len] * window
        # get the magnitude spectrum
        s = np.abs(np.fft.rfft(frame))
        # convert to decibels for perceptual scaling
        s = 20. * np.log10(s + np.finfo(np.float).eps)
        if not prev is None:
            # median positive flux is the novelty value
            novelty[i] = np.median(np.maximum(s - prev, 0))
        prev = s
    # compute a tempogram over the novelty function. We'll skip the first few
    # steps, since they come too close together to perceive distinct events,
    # then return the following 32 values, which is enough to see the basic
    # rhythm pattern over a handful of beats.
    return tempogram(novelty)[4:36]


def combofeats(clip):
    # Compute a feature vector including both rhythm and timbre features.
    frame_len = 1024
    prev_db = None
    num_ceps = 13
    clip_ceps = list()
    novelty = list()
    for s in stft(clip, frame_len):
        # Compute MFCC for this frame.
        s_power = np.square(s) / float(frame_len)
        # apply mel-scaled filter to get mel-band energies for this frame
        mels = np.dot(s_power, mel_fbank.T)
        # take log to account for perceptual nonlinearity
        mels = np.log(mels + np.finfo(np.float).eps)
        # take DCT to convert into cepstrum domain
        frame_ceps = dct(mels, type=2, norm='ortho')
        # should we 'lifter' the ceps here?
        # in any case, skip c0 (total energy) and the higher-order ceps
        clip_ceps.append(frame_ceps[1:1+num_ceps])
        # Compute onset superflux.
        # convert to decibels for perceptual scaling
        s_db = 20. * np.log10(s + np.finfo(np.float).eps)
        if not prev_db is None:
            # median positive flux is the novelty value
            novelty.append(np.median(np.maximum(s_db - prev_db, 0)))
        prev_db = s_db

    rhythm = tempogram(np.array(novelty))[4:36]
    ceps = np.array(clip_ceps)
    cepdiff = np.abs(ceps[:-1,:] - ceps[1:,:])
    feats = (rhythm, ceps.mean(axis=0), ceps.std(axis=0), cepdiff.mean(axis=0))
    return np.concatenate(feats)


def plot_feats(feats, ax):
    # Display distributions of features with a violin plot.
    pos = np.arange(feats.shape[1])
    ax.violinplot(feats, pos, points=400, vert=False, widths=0.9,
            showmeans=False, showextrema=True, showmedians=True)
    # highlight the MFCC features
    ax.axhspan(8-0.2,20+0.2, facecolor='red', alpha=0.15)
    # highlight the chroma features
    ax.axhspan(21-0.2,32+0.2, facecolor='blue', alpha=0.15)


from sklearn.decomposition import PCA
import sklearn.preprocessing


def run(clips):
    # Measure the effectiveness of different features for music similarity
    # calculation.
    feats_A = list()
    feats_B = list()
    for t, clip_A, _, clip_B, _ in clips:
        feats_A.append(combofeats(clip_A))
        feats_B.append(combofeats(clip_B))
    feats_A = np.array(feats_A)
    feats_B = np.array(feats_B)

    # normalize
    summed = np.vstack((feats_A, feats_B))
    summedmean = summed.mean(axis=0)
    summedstd = summed.std(axis=0)
    summed = (summed - summedmean) / summedstd
    feats_A = (feats_A - summedmean) / summedstd
    feats_B = (feats_B - summedmean) / summedstd

    # reduce dimensionality
    pca = PCA(n_components=32)
    pca.fit(summed)
    reduced_A = pca.transform(feats_A)
    reduced_B = pca.transform(feats_B)
    captured = pca.explained_variance_ratio_.sum() * 100
    print "PCA captured %.2f%% of the variance in the library" % captured


    # plot similarity across corresponding track pairs
    plt.set_cmap('hot')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    plt.set_cmap('hot')
    axes[0,0].matshow(np.array(feats_A))
    axes[0,0].set_title("feats_A")
    axes[0,1].matshow(np.array(feats_B))
    axes[0,1].set_title("feats_B")

    axes[1,0].matshow(reduced_A)
    axes[1,0].set_title("PCA reduced feats_A")
    axes[1,1].matshow(reduced_B)
    axes[1,1].set_title("PCA reduced feats_B")

    plt.savefig("features.png", dpi=96, bbox_inches='tight')

