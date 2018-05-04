#!/usr/bin/env python2

import argparse
import config
from scan import scan
from cluster import cluster
import library


def command_init():
    library.init()


def command_clean():
    library.clean()


def command_scan(source):
    scan(source)


def command_cluster():
    cluster()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('init')
    subparsers.add_parser('clean')
    parser_scan = subparsers.add_parser('scan')
    parser_scan.add_argument('source')
    parser_cluster = subparsers.add_parser('cluster')

    kwargs = vars(parser.parse_args())
    config.verbose = kwargs.pop('verbose')
    globals()["command_" + kwargs.pop('command')](**kwargs)
