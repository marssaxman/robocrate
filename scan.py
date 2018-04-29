import audiofile
import thumbnail
import os
import os.path
import sys
from samplerate import resample
import wave
import numpy as np
import uuid
import struct
import config


def _scan_file(source):
    """Extract representative audio thumbnails.

    source: a WAV or other music file readable by ffmpeg
    destination: where we will write output
    Output will be a pair of WAV files with arbitrary names, plus matching
    M3U files, where each M3U contains a path to the original input file.
    """
    print source + ":"
    audio = audiofile.read(source)
    signal = audio.data
    frequency = audio.samplerate
    # Mix down to a single mono channel.
    if hasattr(signal, 'ndim') and signal.ndim > 1:
        print "  mix to mono"
        signal = signal.mean(axis=1).astype(np.float)
    # Skip anything shorter than 2 minutes
    len_sec = len(signal) / float(frequency)
    if len_sec < 120:
        print "  skip short file (%.2f sec)" % len_sec
        return
    # Resample down to 22.1kHz.
    if frequency > 22050.0:
        print "  downsample to 22050 Hz"
        signal = resample(signal, 22050.0 / frequency, 'sinc_fastest')
        frequency = 22050.0
    print "  analyze"
    clip_a, clip_b = thumbnail.get_pair(signal, frequency, size=30.0)
    _write_clip(clip_a, frequency, source)
    _write_clip(clip_b, frequency, source)


def _write_clip(signal, frequency, source):
    if not os.path.isdir(config.dir):
        os.makedirs(config.dir)
    basename = str(uuid.uuid4())
    basepath = os.path.join(config.dir, basename)
    print "  write " + basename + ".m3u"
    with open(basepath + ".m3u", 'w') as fd:
        fd.write(os.path.abspath(source) + os.linesep)
    print "  write " + basename + ".wav"
    data = (signal * np.iinfo(np.int16).max).astype(np.int16)
    wf = wave.open(basepath + ".wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(int(frequency))
    for s in data:
        wf.writeframesraw(struct.pack('<h', s))
    wf.writeframes('')
    wf.close()


def _scan_dir(source):
    print "searching for music files in " + source
    worklist = []
    extensions = tuple(audiofile.list_extensions())
    exclude = set([config.dir])
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(extensions):
                worklist.append(os.path.join(root, file))
    for i, path in enumerate(worklist):
        print "[%d/%d]" % (i+1, len(worklist)),
        try:
            _scan_file(path)
        except (IOError), e:
            print e


def scan(source):
    if os.path.isdir(source):
        _scan_dir(source)
    else:
        _scan_file(source)

