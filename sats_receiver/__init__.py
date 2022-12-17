import pathlib


__version__ = '0.0.1a'


HOMEDIR = pathlib.Path('~/sats_receiver').expanduser()
LOGSDIR = HOMEDIR / 'logs'
TLEDIR = HOMEDIR / 'tle'
