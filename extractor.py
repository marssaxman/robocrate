import os, os.path
import subprocess
from subprocess import PIPE
import tempfile
import json
import sys


def is_present():
    # Is the essentia streaming music extractor present?
    args = ['which', 'essentia_streaming_extractor_music']
    retcode = subprocess.call(args, stdout=PIPE)
    return retcode == 0


def extract(audiofile, jsonfile=None):
    # Run the essentia streaming music extractor. Get its output.
    try:
        temp = None
        if not jsonfile:
            fd, temp = tempfile.mkstemp(prefix="essentia-", suffix=".json")
            os.close(fd)
            jsonfile = temp
        args = ['essentia_streaming_extractor_music', audiofile, jsonfile]
        proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE)
        proc.communicate()

        # the essentia extractor annoyingly returns 1 even if it succeeds
        # to determine whether it actually succeeded, we'll attempt to read
        # the output it was supposed to generate
        with open(jsonfile, 'r') as fd:
            return json.load(fd, strict=False)

    finally:
        if temp:
            os.remove(temp)


def check_present():
    if is_present():
        return
    message = \
    """Cannot find executable 'essentia_streaming_extractor_music'.
    For information about the Essentia extractor, please visit:
        http://essentia.upf.edu/documentation/streaming_extractor_music
    Download the extractor binaries here:
        http://essentia.upf.edu/documentation/extractors"""
    print message
    sys.exit(1)


if __name__ == '__main__':
    check_present()
    info = extract(sys.argv[1])

    def iterkeys(somedict):
        for key, val in somedict.iteritems():
            if isinstance(val, dict):
                for sub in iterkeys(val):
                    yield key + '.' + sub
            elif isinstance(val, list):
                yield "%s[%d]" % (key, len(val))
            else:
                yield key

    for key in iterkeys(info):
        print key

