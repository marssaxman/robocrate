#!/usr/bin/env python2

import argparse
from scan import scan
from cluster import cluster
import library


def command_init(source):
    library.init(source)


def command_clean():
    library.clean()


def command_scan(source=None):
    scan(source)


def command_cluster():
    cluster()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_init = subparsers.add_parser('init')
    parser_init.add_argument('source')

    subparsers.add_parser('clean')

    parser_scan = subparsers.add_parser('scan')
    parser_scan.add_argument('source', nargs='?')

    parser_cluster = subparsers.add_parser('cluster')

    kwargs = vars(parser.parse_args())
    globals()["command_" + kwargs.pop('command')](**kwargs)
