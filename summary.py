# Generate a summary clip from a music recording.
import analysis
import scipy.spatial
import numpy as np


def norm0_1(matrix):
    matrix -= np.min(matrix)
    return matrix / np.max(matrix)


def generate(signal, sample_rate, duration=10.0):
    # Find the most representative section of the source audio to use as its
    # summary. Return an audio clip with the specified duration.
    # Extract feature vector. Create self-similarity matrix.
    # Filter for desirable characteristics. Identify hot spot.
    features = analysis.extract(signal, sample_rate, window=1.0, step=0.5)
    # Normalize the features across all of the vectors.
    means = np.mean(features, axis=1)
    stds = np.std(features, axis=1)
    normals = (features.T - means) / stds
    # Compute pairwise distance between each feature vector. Create a self
    # similarity matrix representing the combinations of distances.
    pairwise_dist = scipy.spatial.distance.pdist(normals, 'cosine')
    similarity = 1.0 - scipy.spatial.distance.squareform(pairwise_dist)
    assert(similarity.shape[0] == similarity.shape[1])
    signal_steps = similarity.shape[0]
    step_rate = (len(signal) / float(sample_rate)) / float(signal_steps)
    similarity = norm0_1(similarity)
    # Filter out columns which are close enough to the beginning or end that
    # we wouldn't have enough room to clip our summary there without extending
    # past the end of the signal. Additionally bias away from any hot spots
    # in the margin within our summary width.
    duration_step = int(duration / step_rate)
    limit = duration_step / 2
    xpos, ypos = np.mgrid[0:signal_steps, 0:signal_steps]
    mask = np.minimum(xpos, ypos)
    mask = np.minimum(mask, np.rot90(mask, k=2))
    similarity *= np.clip((mask - limit) / float(limit), 0, 1)
    # Find the column with the highest median value; that'll be the center of
    # our summary clip. Compute the start and end times, in seconds.
    medians = norm0_1(np.median(similarity, axis=0))
    center_step = medians.argmax()
    clip_start = (center_step - limit) * step_rate
    assert clip_start >= 0
    clip_stop = clip_start + duration
    # Extract the samples from the signal and return.
    idx_start = int(clip_start * sample_rate)
    idx_stop = int(clip_stop * sample_rate)
    return signal[idx_start:idx_stop], sample_rate

