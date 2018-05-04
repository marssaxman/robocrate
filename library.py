import os, os.path
import shutil
import json


dir = os.path.expanduser("~/.robocrate")


class Track(object):
    def __init__(self, **kwargs):
        self._fields = dict(kwargs)

    @property
    def source(self): return self._fields.get('source')

    @property
    def hash(self): return self._fields.get('hash')

    @property
    def summary(self): return self._fields.get('summary')

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

    def save(self):
        filename = self._fields['hash'] + ".json"
        dest = os.path.join(dir, filename)
        with open(dest, 'w') as fp:
            json.dump(self._fields, fp)

    @classmethod
    def create(cls, **kwargs):
        rec = cls(**kwargs)
        rec.save()
        return rec


def tracks():
    # Load all the JSON files in the cache dir. Each one describes one track.
    # Return a list of dicts containing the library info.
    table = list()
    for name in os.listdir(dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(dir, name), 'r') as fd:
            fields = json.load(fd)
            table.append(Track(**fields))
    return table


def clean():
    if not os.path.isdir(dir):
        return
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)


def init():
    if not os.path.isdir(dir):
        os.makedirs(dir)

