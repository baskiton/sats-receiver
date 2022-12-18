#!/usr/bin/env -S python -u

import argparse
import logging
import pathlib

from sats_receiver import utils, HOMEDIR, LOGSDIR, TLEDIR
from sats_receiver.manager import ReceiverManager
from sats_receiver.async_signal import AsyncSignal


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=pathlib.Path)
    ap.add_argument('--log', default='INFO', type=(lambda x: getattr(logging, x.upper(), None)))
    ap.add_argument('--sysu', default=3600, type=int)
    args = ap.parse_args()

    for d in LOGSDIR, TLEDIR:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)

    if not isinstance(args.log, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=args.log, format='%(asctime)s %(levelname)s: %(message)s', filename=LOGSDIR / 'sats_receiver.log')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Hello!')

    asig = AsyncSignal(['SIGABRT', 'SIGHUP', 'SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2'])
    mem = utils.SysUsage(args.sysu)
    mng = ReceiverManager(args.config)

    while not mng.action():
        mem.collect()
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
