import pathlib


HOMEDIR = pathlib.Path('~/sats_receiver').expanduser()
LOGSDIR = HOMEDIR / 'logs'
TLEDIR = HOMEDIR / 'tle'
RECDIR = HOMEDIR / 'records'
SYSRESDIR = pathlib.Path(__file__).absolute().parent / 'systems/resources'
