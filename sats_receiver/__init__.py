import pathlib


HOMEDIR = pathlib.Path('~/sats_receiver').expanduser()
LOGSDIR = HOMEDIR / 'logs'
TLEDIR = HOMEDIR / 'tle'
RECDIR = HOMEDIR / 'records'
