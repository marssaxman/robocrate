# Generate a summary clip from a music recording.
import analysis
import scipy.spatial
import numpy as np


def _norm0_1(matrix):
    matrix -= np.min(matrix)
    return matrix / np.max(matrix)


def _analyze(signal, sample_rate):
    features = analysis.extract(signal, sample_rate, window=1.0, step=0.5)
    # Normalize the features across all of the vectors.
    means = np.mean(features, axis=1)
    stds = np.std(features, axis=1)
    return (features.T - means) / stds


def _similarity_matrix(features):
    # Compute pairwise distance between each feature vector. Create a self
    # similarity matrix representing the combinations of distances.
    pairwise_dist = scipy.spatial.distance.pdist(features, 'cosine')
    matrix = 1.0 - scipy.spatial.distance.squareform(pairwise_dist)
    assert(matrix.shape[0] == matrix.shape[1])
    return _norm0_1(matrix)


def _protect_edges(similarity, duration, step_rate):
    # Filter out columns which are close enough to the beginning or end that
    # we wouldn't have enough room to clip our summary there without extending
    # past the end of the signal. Additionally bias away from any hot spots
    # in the margin within our summary width.
    limit = int(duration / step_rate) / 2
    xpos, ypos = np.mgrid[0:similarity.shape[0], 0:similarity.shape[1]]
    mask = np.minimum(xpos, ypos)
    mask = np.minimum(mask, np.rot90(mask, k=2))
    return similarity * np.clip((mask - limit) / float(limit), 0, 1)


def generate(signal, sample_rate, duration=10.0):
    # Find the most representative section of the source audio to use as its
    # summary. Return an audio clip with the specified duration.
    # Extract feature vector. Create self-similarity matrix.
    # Filter for desirable characteristics. Identify hot spot.
    features = _analyze(signal, sample_rate)
    similarity = _similarity_matrix(features)
    signal_steps = similarity.shape[0]
    step_rate = (len(signal) / float(sample_rate)) / float(signal_steps)
    similarity = _protect_edges(similarity, duration, step_rate)
    # Find the column with the highest median value; that'll be the center of
    # our summary clip. Compute the start and end times, in seconds.
    medians = _norm0_1(np.median(similarity, axis=0))
    center_step = medians.argmax()
    clip_start = (center_step - duration/2.0) * step_rate
    clip_stop = clip_start + duration
    assert clip_start >= 0 and clip_stop <= signal_steps
    # Extract the samples from the signal and return.
    idx_start = int(clip_start * sample_rate)
    idx_stop = int(clip_stop * sample_rate)
    assert idx_start >= 0 and idx_stop <= len(signal)
    return signal[idx_start:idx_stop], sample_rate

