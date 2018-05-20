import os.path
import json
import numpy as np

# Extract numeric features from an Essentia audio feature report.
# Save them as a numpy array of floats.
# We must decide how many features there are and what order they come in.

_feature_paths = []
_feature_names = []
def init():
    global _feature_paths
    global _feature_names
    assert len(_feature_paths) == len(_feature_names)
    if len(_feature_paths) != 0:
        return
    stats = [
        "min", "max", "median", "mean", "var", "stdev",
        "dmean", "dvar", "dmean2", "dvar2"
    ]
    all_features = [
        ("lowlevel", [
            "average_loudness",
            # (("barkbands", stats), 27),
            ("barkbands_crest", stats),
            ("barkbands_flatness_db", stats),
            ("barkbands_kurtosis", stats),
            ("barkbands_skewness", stats),
            ("barkbands_spread", stats),
            ("dissonance", stats),
            "dynamic_complexity",
            # (("erbbands", stats), 40),
            ("erbbands_crest", stats),
            ("erbbands_flatness_db", stats),
            ("erbbands_kurtosis", stats),
            ("erbbands_skewness", stats),
            ("erbbands_spread", stats),
            ("hfc", stats),
            ("loudness_ebu128", [
                "integrated",
                "loudness_range",
                ("momentary", stats),
                ("short_term", stats),
            ]),
            # (("melbands", stats), 40),
            # (("melbands128", stats), 128),
            ("melbands_crest", stats),
            ("melbands_flatness_db", stats),
            ("melbands_kurtosis", stats),
            ("melbands_skewness", stats),
            ("melbands_spread", stats),
            ("pitch_salience", stats),
            ("silence_rate_20dB", stats),
            ("silence_rate_30dB", stats),
            ("silence_rate_60dB", stats),
            ("spectral_centroid", stats),
            ("spectral_complexity", stats),
            (("spectral_contrast_coeffs", stats), 6),
            (("spectral_contrast_valleys", stats), 6),
            ("spectral_decrease", stats),
            ("spectral_energy", stats),
            ("spectral_energyband_high", stats),
            ("spectral_energyband_low", stats),
            ("spectral_energyband_middle_high", stats),
            ("spectral_energyband_middle_low", stats),
            ("spectral_entropy", stats),
            ("spectral_flux", stats),
            ("spectral_kurtosis", stats),
            ("spectral_rms", stats),
            ("spectral_rolloff", stats),
            ("spectral_skewness", stats),
            ("spectral_spread", stats),
            ("spectral_strongpeak", stats),
            ("zerocrossingrate", stats),
            ("gfcc", [
                ("mean", 13),
                # (("cov", 13), 13),
                # (("icov", 13), 13),
            ]),
            ("mfcc", [
                ("mean", 13)
                # (("cov", 13), 13),
                # (("icov", 13), 13),
            ]),
        ]),
        ("rhythm", [
            ("beats_loudness", stats),
            "bpm",
            # ("bpm_histogram, 250),
            "bpm_histogram_first_peak_bpm",
            "bpm_histogram_first_peak_weight",
            "bpm_histogram_second_peak_bpm",
            "bpm_histogram_second_peak_weight",
            "danceability",
            "onset_rate",
            (("beats_loudness_band_ratio", stats), 6),
        ]),
        ("tonal", [
            "chords_changes_rate",
            "chords_number_rate",
            ("chords_strength", stats),
            ("hpcp_crest", stats),
            ("hpcp_entropy", stats),
            "tuning_diatonic_strength",
            "tuning_equal_tempered_deviation",
            "tuning_frequency",
            "tuning_nontempered_energy_ratio",
            (("hpcp", stats), 36),
            (("chords_histogram"), 24),
            ("thpcp", 36),
        ]),
    ]
    # Unpack that schema. We will end up with a list of lists. Each list
    # will be the path to exactly one feature. We will then generate a matching
    # list of feature names.
    def flatten(obj):
        if isinstance(obj, int):
            for i in range(obj):
                yield (i, )
        elif isinstance(obj, str) or isinstance(obj, unicode):
            yield (obj, )
        elif isinstance(obj, tuple):
            outer, inner = obj
            for o in flatten(outer):
                for i in flatten(inner):
                    yield o + i
        else:
            for item in obj:
                for i in flatten(item):
                    yield i
    _feature_paths = list(flatten(all_features))
    _feature_names = [".".join(str(f) for f in p) for p in _feature_paths]


def _follow(obj, key, *args):
    val = obj[key]
    return _follow(val, *args) if len(args) else val


def extract(details, track):
    # Extract the features we care about from the Essentia json object.
    # Save them into a Numpy array. Write that array to the 'features' path.
    init()
    array = np.zeros(len(_feature_paths), dtype=np.float)
    for i, path in enumerate(_feature_paths):
        array[i] = _follow(details, *path)
    np.save(track.features_file, array)


def check(track):
    if not os.path.isfile(track.features_file):
        return False
    try:
        init()
        data = np.load(track.features_file)
        return data.shape == (len(_feature_paths),)
    except IOError as e:
        return False


def generate(track):
    with open(track.details_file, 'r') as fd:
        extract(json.load(fd, strict=False), track)


if __name__ == '__main__':
    init()
    print "Saved features array schema:"
    for i in enumerate(_feature_names):
        print "    %d: '%s'" % i
    print "Total features: %d" % len(_feature_names)
