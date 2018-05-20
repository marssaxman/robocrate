import os, os.path, sys
from musictoys import audiofile
from mp3hash import mp3hash
import eyed3
import random

import summary
import library
import extractor


def _scan_file(source):
    """Extract representative audio summary segments and generate metadata.

    source: an MP3, WAV, or other music file readable by ffmpeg
    """
    # Generate the summary clip, if it doesn't already exist.
    info = {
        "source": os.path.abspath(source),
        "hash": mp3hash(source),
    }

    # Add ID3 metadata, if available.
    try:
        eyed3.log.setLevel("ERROR")
        id3file = eyed3.load(source)
        if id3file and id3file.tag:
            tag = id3file.tag
            if tag.title:
                info["title"] = tag.title
            if tag.artist:
                info["artist"] = tag.artist
            if tag.genre:
                info["genre"] = tag.genre.name
            if tag.bpm:
                info["bpm"] = tag.bpm
            release_date = tag.best_release_date
            if release_date:
                info["year"] = release_date.year
    except UnicodeDecodeError:
        pass

    # Insert the track record into our library.
    track = library.Track.create(info)


def _search(source):
    print "searching for music files in " + source
    worklist = []
    extensions = tuple(audiofile.extensions())
    exclude = set([library.DIR])
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if not file.endswith(extensions):
                continue
            worklist.append(os.path.join(root, file))
    return worklist


def _filter_known(worklist):
    # Remove all the files which are already present in our track library.
    known = set()
    for info in library.tracks():
        if not info.source:
            continue
        known.add(info.source)
    abslist = (os.path.abspath(p) for p in worklist)
    return [p for p in abslist if not p in known]


def attempt(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except KeyboardInterrupt:
        sys.exit(0)
    except IOError as e:
        print "  failed: %s" % str(e)


def scan(source=None):

    if source is None:
        basedir = library.source()
        worklist = _search(basedir)
    elif os.path.isdir(source):
        basedir = source
        worklist = _search(source)
    else:
        basedir = os.getcwd()
        worklist = [source]

    # Update the library track list.
    if len(worklist):
        random.shuffle(worklist)
        print "Updating track library"
        worklist = _filter_known(worklist)
    for i, path in enumerate(worklist):
        relpath = os.path.relpath(path, basedir)
        printpath = relpath if len(relpath) < len(path) else path
        print "[%d/%d] %s" % (i+1, len(worklist), printpath)
        attempt(_scan_file, path)

    # If there are tracks in the library with no details, go analyze them.
    worklist = [t for t in library.tracks() if not os.path.isfile(t.details)]
    if len(worklist):
        random.shuffle(worklist)
        print "Extracting music information"
    for i, track in enumerate(worklist):
        print "[%d/%d] %s" % (i+1, len(worklist), track.caption)
        attempt(extractor.extract, track.source, track.details)

    # If there are files in the library which are missing their summaries,
    # go generate summary clips.
    worklist = [t for t in library.tracks() if not os.path.isfile(t.summary)]
    if len(worklist):
        random.shuffle(worklist)
        print "Generating summary clips"
    for i, track in enumerate(worklist):
        print "[%d/%d] %s" % (i+1, len(worklist), track.caption)
        attempt(summary.generate, track.source, track.summary)

    # future: extract relevant features from essentia details and save as
    # numpy file; this will save a lot of JSON-parsing time

