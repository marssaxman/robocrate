import os
#import json
#from tempfile import NamedTemporaryFile

dir = os.path.expanduser("~/.robocrate")

#_catalog = {}
#{
#    "tracks" : [
#        {
#            "file" : <string>,
#            "mtime" : <string>,
#            "thumbs" : [ <string> ]
#        }
#    ]
#}


def init():
    if not os.path.isdir(dir):
        os.makedirs(dir)


#def load():
#    if os.path.isfile(path):
#        with open(path, 'r') as fd:
#            _catalog = json.load(fd)


#def store():
#    with tempfile.NamedTemporaryFile('w', dir=dir, delete=False) as tf:
#        json.dump(_catalog, tf)
#        tempname = tf.name
#        os.rename(tempname, path)


