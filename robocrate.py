#!/usr/bin/env python2

import os
import argparse
import shutil
import config
from scan import scan


def command_init():
    if not os.path.isdir(config.dir):
        os.makedirs(config.dir)


def command_clean():
    if not os.path.isdir(config.dir):
        return
    for name in os.listdir(config.dir):
        path = os.path.join(config.dir, name)
        if os.path.isfile(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)


def command_scan(source):
    scan(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('init')
    subparsers.add_parser('clean')
    parser_scan = subparsers.add_parser('scan')
    parser_scan.add_argument('source')

    kwargs = vars(parser.parse_args())
    config.verbose = kwargs.pop('verbose')
    globals()["command_" + kwargs.pop('command')](**kwargs)

