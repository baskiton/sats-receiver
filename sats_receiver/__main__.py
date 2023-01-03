#!/usr/bin/env -S python -u

import argparse
import logging
import pathlib

from sats_receiver import HOMEDIR, LOGSDIR, TLEDIR, RECDIR
from sats_receiver.manager import ReceiverManager
from sats_receiver.async_signal import AsyncSignal


def setup_logging(log_lvl):
    if not isinstance(log_lvl, int):
        raise ValueError('Invalid log level: %s' % log_lvl)

    fmt = '%(asctime)s %(levelname)s: %(name)s: %(message)s'
    logging.basicConfig(level=log_lvl, format=fmt, filename=LOGSDIR / 'sats_receiver.log')
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(sh)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=pathlib.Path)
    ap.add_argument('--log', default='INFO', type=(lambda x: getattr(logging, x.upper(), None)))
    ap.add_argument('--sysu', default=3600, type=int)
    args = ap.parse_args()

    for d in LOGSDIR, TLEDIR, RECDIR:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)

    setup_logging(args.log)

    logging.info('Hello!')

    asig = AsyncSignal(['SIGABRT', 'SIGHUP', 'SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2'])
    mng = ReceiverManager(args.config, args.sysu)

    while not mng.action():
        signame = asig.wait(1)
        if signame:
            if 'USR' in signame:
                # TODO
                pass
            else:
                mng.stop()
                logging.info('Exit by %s', signame)
                break

    mng.wait()
    logging.info('Bye!')
