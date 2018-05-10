import os
import os.path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def hamming(N):
    # improved hamming window: original implementation used 0.54, 0.46
    i = np.arange(N).astype(np.float)
    return 0.53836 - (0.46164 * np.cos(np.pi * 2.0 * i / (N-1)))


def stft(clip, frame_len=1024):
    # generator for magnitude spectrum frames from an audio clip
    # assume clip is 22050Hz mono; normalize it
    clip = clip.astype(np.float)
    clip -= clip.mean()
    clip /= np.abs(clip).max()
    # assume half-overlap
    step_len = frame_len
    #step_len = int(frame_len / 2)
    window = hamming(frame_len)
    num_frames = int(np.floor((len(clip) - frame_len + 1) / step_len))
    for i in xrange(num_frames):
        offset = i * step_len
        # get the samples and multiply by the window to reduce ringing
        frame = clip[offset:offset+frame_len] * window
        # get the magnitude spectrum
        s = np.abs(np.fft.rfft(frame))
        yield s


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
    frame_len = 512
    prev_db = None
    novelty = list()
    for s in stft(clip, frame_len):
        # convert to decibels for perceptual scaling
        s_db = 20. * np.log10(s + np.finfo(np.float).eps)
        if not prev_db is None:
            # median positive flux is the novelty value
            novelty.append(np.median(np.maximum(s_db - prev_db, 0)))
        prev_db = s_db
    novelty = np.array(novelty)
    spec = np.abs(np.fft.rfft(novelty))[1:]
    gram = np.argsort(spec)[::-1][:6] / float(len(spec))
    return gram


def run(clips):
    feats = list()
    print "analyzing clips"
    for t, clip_A, _, clip_B, _ in clips:
        feats.append(rhythmfeats(clip_A))
        feats.append(rhythmfeats(clip_B))

    plt.set_cmap('hot')
    plt.matshow(feats)
    plt.savefig("tempogram.png", dpi=96, bbox_inches='tight')

