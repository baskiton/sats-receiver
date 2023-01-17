import pathlib


__version__ = '0.1.0'


HOMEDIR = pathlib.Path('~/sats_receiver').expanduser()
LOGSDIR = HOMEDIR / 'logs'
TLEDIR = HOMEDIR / 'tle'
RECDIR = HOMEDIR / 'records'
