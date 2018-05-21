import os, os.path, sys
from mp3hash import mp3hash
import eyed3
import random

import library
#import summary
import extractor
import features


def _scan_file(source):
    """Extract metadata from this audio file."""
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
            if tag.album_artist:
                info["album_artist"] = tag.album_artist
            if tag.genre:
                info["genre"] = tag.genre.name
            if tag.bpm:
                info["bpm"] = tag.bpm
            release_date = tag.best_release_date
            if release_date:
                info["year"] = release_date.year
            if tag.publisher:
                info["publisher"] = tag.publisher
    except UnicodeDecodeError:
        pass

    # Insert the track record into our library.
    track = library.Track.create(info)


def _search(source):
    print "searching for music files in " + source
    worklist = []
    extensions = ('.aac', '.aiff', '.au', '.flac', '.m4a', '.m4r',
            '.mp2', '.mp3', '.mp4', '.ogg', '.oga', '.wav', '.wma')
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


def process(module, label):
    worklist = [t for t in library.tracks() if not module.check(t)]
    if len(worklist):
        random.shuffle(worklist)
        print label
    for i, track in enumerate(worklist):
        print "[%d/%d] %s" % (i+1, len(worklist), track.caption)
        attempt(module.generate, track)


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
    process(extractor, "Extracting music information")
    process(features, "Harvesting feature matrix")
    #process(summary, "Generating summary clips")

