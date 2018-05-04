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


def generate(signal, sample_rate, duration=10.0):
    signal_time = len(signal) / float(sample_rate)
    # Find the most representative section of the source audio to use as its
    # summary. Return an audio clip with the specified duration.
    # Extract feature vector. Create self-similarity matrix.
    # Filter for desirable characteristics. Identify hot spot.
    features = _analyze(signal, sample_rate)
    similarity = _similarity_matrix(features)
    signal_steps = similarity.shape[0]
    step_rate = float(signal_steps) / signal_time
    clip_steps = int(duration * step_rate)
    # Find the column with the highest median value; that'll be the center of
    # our summary clip.
    medians = _norm0_1(np.median(similarity, axis=0))
    limit_steps = clip_steps / 2
    start_step = medians[limit_steps:-limit_steps].argmax()
    stop_step = start_step + clip_steps
    # Extract the samples from the signal and return.
    start_idx = int(start_step / step_rate * sample_rate)
    stop_idx = int(stop_step / step_rate * sample_rate)
    return signal[start_idx:stop_idx]

