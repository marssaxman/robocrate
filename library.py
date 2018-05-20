import os, os.path
import sys
import shutil
import json


DIR = os.path.expanduser("~/.robocrate")
LIBRARY = os.path.join(DIR, "library.json")

_library = None
_tracklist = None


class Track(object):
    def __init__(self, fields):
        self._fields = fields

    @property
    def source(self): return self._fields.get('source')

    @property
    def hash(self): return self._fields.get('hash')

    @property
    def title(self): return self._fields.get('title')

    @property
    def artist(self): return self._fields.get('artist')

    @property
    def genre(self): return self._fields.get('genre')

    @property
    def bpm(self): return self._fields.get('bpm')

    @property
    def year(self): return self._fields.get('year')

    @property
    def summary_file(self):
        return os.path.join(DIR, self.hash + '.wav')

    @property
    def details_file(self):
        return os.path.join(DIR, self.hash + '.json')

    @property
    def features_file(self):
        return os.path.join(DIR, self.hash + '.npy')

    @property
    def caption(self):
        if self.title and self.artist:
            return "%s - %s" % (self.artist, self.title)
        if self.title:
            return self.title
        return os.path.splitext(os.path.basename(self.source))[0]

    def save(self):
        tracks().save()

    @staticmethod
    def create(fields):
        assert fields['source']
        assert fields['hash']
        return tracks().insert(fields)


class Tracklist(object):
    def __init__(self, track_dicts):
        self._track_dicts = track_dicts
        self._track_objs = [Track(t) for t in track_dicts]

    def __len__(self):
        return len(self._track_objs)

    def __getitem__(self, index):
        return self._track_objs[index]

    def __iter__(self):
        for t in self._track_objs:
            yield t

    def save(self):
        commit()

    def insert(self, fields):
        self._track_dicts.append(fields)
        t = Track(fields)
        self._track_objs.append(t)
        commit()
        return t


def load():
    global _library
    if not _library is None:
        return
    if not os.path.isdir(DIR):
        sys.stderr.write(
            "Cannot find ~/.robocrate directory; must 'init' first.\n")
        sys.exit(1)
    if not os.path.isfile(LIBRARY):
        sys.stderr.write(
            "Cannot find ~/.robocrate/config.json file; must 'init' first.\n")
        sys.exit(1)
    with open(LIBRARY, 'r') as fd:
        _library = json.load(fd)


def commit():
    global _library
    with open(LIBRARY, 'w') as fd:
        json.dump(_library, fd)


def source():
    global _library
    load()
    return _library['source']


def tracks():
    global _library
    global _tracklist
    if _tracklist is None:
        load()
        _tracklist = Tracklist(_library['tracks'])
    return _tracklist


def clean():
    if not os.path.isdir(DIR):
        return
    if os.path.isfile(LIBRARY):
        load()

    # Make a list of the files we expect to find in the library.
    # We'll delete everything we don't recognize.
    expected = {LIBRARY}
    for track in tracks():
        if track.summary_file:
            expected.add(os.path.abspath(track.summary_file))
        if track.details_file:
            expected.add(os.path.abspath(track.details_file))
        if track.features_file:
            expected.add(os.path.abspath(track.features_file))

    for name in os.listdir(DIR):
        path = os.path.abspath(os.path.join(DIR, name))
        if path in expected:
            continue
        if os.path.isfile(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)


def init(source):

    if os.path.isfile(source):
        sys.stderr.write("Source path '%s' is a file, not a directory.\n")
        sys.exit(1)
    if not os.path.isdir(source):
        sys.stderr.write("Source directory '%s' does not exist.\n")
        sys.exit(1)

    if not os.path.isdir(DIR):
        os.makedirs(DIR)

    global _library
    _library = {
        "source": source,
        "tracks": [],
    }

    commit()

