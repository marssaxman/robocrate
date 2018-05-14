import os, os.path, sys
import musictoys
from musictoys import audiofile
from mp3hash import mp3hash
import eyed3
import numpy as np
import struct
import random

import summary
import library


def _gen_summary(source, dest):
    # Read the audio data.
    signal, samplerate = audiofile.read(source)
    # Normalize to mono 22k for consistent analysis.
    signal, samplerate = musictoys.analysis.normalize(signal, samplerate)
    # Find the most representative 30 seconds to use as a summary clip.
    print "  analyze"
    clip = summary.generate(signal, samplerate, duration=30.0)
    # Write the summary as a 16-bit WAV.
    print "  write summary"
    audiofile.write(dest, clip, samplerate)


def _scan_file(source):
    """Extract representative audio summary segments and generate metadata.

    source: an MP3, WAV, or other music file readable by ffmpeg
    """
    hash = mp3hash(source)
    base_path = os.path.join(library.dir, hash)

    # Generate the summary clip, if it doesn't already exist.
    summary_path = base_path + '.wav'
    if not os.path.isfile(summary_path):
        _gen_summary(source, summary_path)

    info = {
        "source": os.path.abspath(source),
        "hash": hash,
        "summary": os.path.abspath(summary_path),
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
    library.Track.create(**info)


def _search(source):
    print "searching for music files in " + source
    worklist = []
    extensions = tuple(audiofile.extensions())
    exclude = set([library.dir])
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
        # We should have generated a summary clip for this track, and it should
        # still exist where we expect it.
        if not info.summary:
            continue
        if not os.path.isfile(info.summary):
            continue
        if not info.source:
            continue
        known.add(info.source)
    abslist = (os.path.abspath(p) for p in worklist)
    return [p for p in abslist if not p in known]


def scan(source):
    if os.path.isdir(source):
        basedir = source
        worklist = _search(source)
    else:
        basedir = os.getcwd()
        worklist = [source]
    random.shuffle(worklist)
    print "skipping known files"
    worklist = _filter_known(worklist)
    if not os.path.isdir(library.dir):
        os.makedirs(library.dir)
    for i, path in enumerate(worklist):
        relpath = os.path.relpath(path, basedir)
        printpath = relpath if len(relpath) < len(path) else path
        print "[%d/%d] %s" % (i+1, len(worklist), printpath)
        try:
            _scan_file(path)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print "  failed: %s" % str(e)
            pass

