# Extract representative thumbnails from an audio signal.
import numpy
import scipy
import scipy.signal
import features


def _self_similarity_matrix(featureVectors):
    from scipy.spatial import distance
    [nDims, nVectors] = featureVectors.shape
    [featureVectors2, MEAN, STD] = features.normalize([featureVectors.T])
    featureVectors2 = featureVectors2[0].T
    return 1.0 - distance.squareform(distance.pdist(featureVectors2.T, 'cosine'))


def find_pair(signal, frequency, size=10.0, window=1.0, step=0.5):
    Limit1 = 0
    Limit2 = 1
    # Compute the features we will use to measure similarity.
    vectors = features.extract(signal, frequency, window, step)
    # Create the diagonal matrix which lets us find self-similar regions.
    similarity = _self_similarity_matrix(vectors)

    # Apply a moving filter.
    M = int(round(size / step))
    B = numpy.eye(M, M)
    similarity = scipy.signal.convolve2d(similarity, B, 'valid')
    shape = similarity.shape

    # Remove main diagonal elements as a post-processing step.
    minVal = numpy.min(similarity)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(i-j) < 5.0 / step or i > j:
                similarity[i, j] = minVal

    # find the maximum position
    similarity[0:int(Limit1*shape[0]), :] = minVal
    similarity[:, 0:int(Limit1*shape[0])] = minVal
    similarity[int(Limit2*shape[0])::, :] = minVal
    similarity[:, int(Limit2*shape[0])::] = minVal

    maxVal = numpy.max(similarity)
    [I, J] = numpy.unravel_index(similarity.argmax(), shape)
    i1, i2 = I, I
    j1, j2 = J, J

    while i2-i1 < M:
        if i1 <= 0 or j1 <= 0 or i2 >= shape[0]-2 or j2 >= shape[1]-2:
            break
        if similarity[i1-1, j1-1] > similarity[i2+1, j2+1]:
            i1 -= 1
            j1 -= 1
        else:
            i2 += 1
            j2 += 1

    return ((step*i1, step*i2), (step*j1, step*j2))


def get_pair(signal, frequency, **kwargs):
    ((i_lo, i_hi), (j_lo, j_hi)) = find_pair(signal, frequency, **kwargs)
    clip_i = signal[int(frequency * i_lo):int(frequency * i_hi)]
    clip_j = signal[int(frequency * j_lo):int(frequency * j_hi)]
    return clip_i, clip_j
