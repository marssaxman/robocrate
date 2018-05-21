#!/usr/bin/env python

import argparse
import scan
import cluster
import library
import train
import extractor


def command_init(*args, **kwargs):
    library.init(*args, **kwargs)
    extractor.init()


def command_clean(*args, **kwargs):
    library.clean(*args, **kwargs)


def command_scan(*args, **kwargs):
    extractor.init()
    scan.scan(*args, **kwargs)


def command_cluster(*args, **kwargs):
    cluster.cluster(*args, **kwargs)


def command_train(*args, **kwargs):
    train.train(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_init = subparsers.add_parser('init')
    parser_init.add_argument('source')

    subparsers.add_parser('clean')
    scan.add_arguments(subparsers.add_parser('scan'))
    subparsers.add_parser('cluster')
    train.add_arguments(subparsers.add_parser('train'))

    kwargs = vars(parser.parse_args())
    globals()["command_" + kwargs.pop('command')](**kwargs)
