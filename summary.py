# Generate a summary clip from a music recording.
import os.path
import analysis
import scipy.spatial
import numpy as np
import musictoys.analysis
from musictoys import audiofile


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


def extract(signal, duration=10.0):
    signal_time = len(signal) / float(signal.sample_rate)
    # Find the most representative section of the source audio to use as its
    # summary. Return an audio clip with the specified duration.
    # Extract feature vector. Create self-similarity matrix.
    # Filter for desirable characteristics. Identify hot spot.
    features = _analyze(signal, sample_rate)
    similarity = _similarity_matrix(features)
    step_rate = float(similarity.shape[0]) / signal_time
    clip_steps = int(duration * step_rate)
    # Score each column by taking the median of all row values.
    score = _norm0_1(np.median(similarity, axis=0))
    # Compute the start time with the highest average score across the window.
    starts = np.zeros(len(score)-clip_steps)
    for i in range(len(starts)):
        starts[i] = np.mean(score[i:i+clip_steps])
    start_step = starts.argmax()
    stop_step = start_step + clip_steps
    # Extract the samples from the signal and return.
    start_idx = int(start_step / step_rate * sample_rate)
    stop_idx = int(stop_step / step_rate * sample_rate)
    return signal[start_idx:stop_idx]


def check(track):
    return os.path.isfile(track.summary)


def generate(track):
    # Read the audio data.
    signal = audiofile.read(track.source)
    # Normalize to mono 22k for consistent analysis.
    signal = musictoys.analysis.normalize(signal)
    # Find the most representative 30 seconds to use as a summary clip.
    print "  analyze"
    clip = extract(signal, duration=30.0)
    # Write the summary as a 16-bit WAV.
    print "  write summary"
    audiofile.write(track.summary_file, clip)

