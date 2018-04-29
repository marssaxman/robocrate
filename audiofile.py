# Simple interface for reading and writing sound files in a variety of formats.
# Uses pysndfile if possible; falls back to ffmpeg if necessary.

import pysndfile
from collections import namedtuple
import warnings
import os
import subprocess
import tempfile


AudioFile = namedtuple('AudioFile', 'data samplerate format')


def _read_pysndfile(filename):
    f = pysndfile.PySndfile(filename)
    nframes = f.frames()
    data = f.read_frames(nframes)
    return AudioFile(data, f.samplerate(), f.format())


def _popen(cmd):
    pipe = subprocess.PIPE
    return subprocess.Popen(cmd, stdin=pipe, stdout=pipe, stderr=pipe)


def _has_ffmpeg():
    # try to run ffmpeg -version and see if we get a sane result.
    proc = _popen(['ffmpeg', '-version'])
    proc.communicate()
    return 0 == proc.returncode


def _read_ffmpeg(filename):
    # use ffmpeg to convert the input file to a temporary wav file we can read
    try:
        tempfd, temppath = tempfile.mkstemp(suffix='.wav')
        proc = _popen(['ffmpeg', '-v', '1', '-y', '-i', filename, temppath])
        proc.communicate()
        if proc.returncode:
            return None
        return _read_pysndfile(temppath)
    finally:
        os.close(tempfd)
        os.remove(temppath)


# This will be populated by the first call to list_extensions().
_supported_extensions = None


def list_extensions():
    global _supported_extensions
    if _supported_extensions is None:
        with warnings.catch_warnings():
            # pysndfile likes to complain about formats that libsndfile
            # supports, but which haven't been added to pysndfile itself yet;
            # we really don't care and would rather not be pestered about it.
            warnings.simplefilter('ignore')
            _supported_extensions = pysndfile.get_sndfile_formats()
        if _has_ffmpeg():
            _supported_extensions.append('mp3')
    return _supported_extensions


def read(filename):
    # Try to read the file with pysndfile. If it fails, try ffmpeg.
    try:
        return _read_pysndfile(filename)
    except IOError:
        f = _read_ffmpeg(filename)
        if f is None:
            raise
        return f


def write(filename, contents):
    channels = contents.data.ndim
    format = contents.format
    f = pysndfile.PySndfile(
        filename,
        mode='w',
        format=format,
        channels=contents.data.ndim,
        samplerate=contents.samplerate
    )
    f.write_frames(contents.data)
