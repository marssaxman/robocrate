import os
import os.path
import subprocess
from subprocess import PIPE
import tempfile
import json
import sys
import features


# NOTE: streaming_extractor_music has a dependency on 'libav'.
# not sure how to test for that.
# ooooh, statically linked versions available:
#   http://acousticbrainz.org/download


def is_present():
    # Is the essentia streaming music extractor present?
    args = ['which', 'streaming_extractor_music']
    retcode = subprocess.call(args, stdout=PIPE)
    return retcode == 0


def check(track):
    return os.path.isfile(track.details_file)


def update_tags(blob, track):
    # See if there's any interesting metadata we can harvest.
    metadata = blob.get('metadata')
    if not metadata:
        return
    tags = metadata.get('tags')
    if not tags:
        return
    update = dict()
    album = tags.get('album')
    if album:
        update['album'] = album[0]
    artist = tags.get('artist')
    if artist:
        update['artist'] = artist[0]
    title = tags.get('title')
    if title:
        update['title'] = title[0]
    genre = tags.get('genre')
    if genre:
        update['genre'] = genre[0]
    label = tags.get('label')
    if label:
        update['publisher'] = label[0]
    remixer = tags.get('remixer')
    if remixer:
        update['remixer'] = remixer[0]
    if len(update):
        track.update(update)


def generate(track):
    blob = extract(track.source, track.details_file)
    features.extract(blob, track)
    update_tags(blob, track)


def extract(audiofile, jsonfile=None):
    # Run the essentia streaming music extractor. Get its output.
    try:
        temp = None
        if not jsonfile:
            fd, temp = tempfile.mkstemp(prefix="essentia-", suffix=".json")
            os.close(fd)
            jsonfile = temp
        args = ['streaming_extractor_music', audiofile, jsonfile]
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
        """Cannot find executable 'streaming_extractor_music'.
    For information about the Essentia extractor, please visit:
        http://essentia.upf.edu/documentation/streaming_extractor_music
    Download the extractor binaries here:
        http://essentia.upf.edu/documentation/extractors"""
    print message
    sys.exit(1)


if __name__ == '__main__':
    check_present()
    info = extract(sys.argv[1])

    def printout(val, tabs):
        if isinstance(val, dict):
            if "mean" in val and "dmean" in val:
                if isinstance(val["mean"], list):
                    print "STATS [%d]" % len(val["mean"])
                else:
                    print "STATS"
                return
            else:
                print
            tabs += "    "
            for key in sorted(list(val.keys())):
                sub = val[key]
                print "%s%s:" % (tabs, key),
                printout(sub, tabs)
        elif isinstance(val, list):
            print "[%d]" % len(val),
            printout(val[0], tabs + "    ")
        else:
            name = str(type(val))
            print {
                "<type 'float'>": "float",
                "<type 'unicode'>": "string",
                "<type 'int'>": "int",
            }.get(name, name)

    printout(info, "")
