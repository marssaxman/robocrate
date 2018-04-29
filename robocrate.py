import argparse
import config
import scan


def command_init():
    config.init()


def command_scan(source):
    scan.scan(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_init = subparsers.add_parser('init')

    parser_scan = subparsers.add_parser('scan')
    parser_scan.add_argument('source')

    kwargs = vars(parser.parse_args())
    globals()["command_" + kwargs.pop('command')](**kwargs)

