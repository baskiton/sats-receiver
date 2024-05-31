#!/usr/bin/env -S python -u

import argparse
import atexit
import logging
import logging.handlers
import multiprocessing as mp
import pathlib
import sys

import gnuradio as gr
import gnuradio.gr

from sats_receiver import HOMEDIR, LOGSDIR, TLEDIR, RECDIR
from sats_receiver.async_signal import AsyncSignal
from sats_receiver.manager import ReceiverManager
from sats_receiver.utils import SysUsage


def setup_logging(q: mp.Queue, log_lvl: int):
    if not isinstance(log_lvl, int):
        raise ValueError('Invalid log level: %s' % log_lvl)

    logger = logging.getLogger()
    logger.setLevel(log_lvl)
    logger.addHandler(logging.handlers.QueueHandler(q))
    mp.get_logger().setLevel(log_lvl)

    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(name)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.handlers.TimedRotatingFileHandler(LOGSDIR / 'sats_receiver.log', 'midnight')
    fh.setFormatter(fmt)

    qhl = logging.handlers.QueueListener(q, sh, fh)
    qhl.start()
    atexit.register(qhl.stop)

    # PIL logging level
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.DEBUG + 2)

    gr_logger = gr.gr.logging()
    gr_logger.set_default_level(gr.gr.log_levels.info)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=pathlib.Path, help='Config file path')
    ap.add_argument('--exec', type=pathlib.Path,
                    help='Python script containing specified executor named `Executor`')
    ap.add_argument('--exec_config', type=pathlib.Path,
                    help='Executor specific config file path')
    ap.add_argument('--log', default='INFO', type=(lambda x: getattr(logging, x.upper(), None)),
                    help='Logging level, INFO default')
    ap.add_argument('--sysu', default=SysUsage.DEFAULT_INTV, type=int,
                    help='System Usages info timeout in seconds, 1 hour default')
    args = ap.parse_args()

    for d in LOGSDIR, TLEDIR, RECDIR:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)

    q = mp.Queue()
    setup_logging(q, args.log)

    logging.info('Hello!')

    kw = dict(sysu_intv=args.sysu)
    if args.exec:
        sys.path.append(str(args.exec.expanduser().absolute().parent))
        kw['executor_cls'] = __import__(args.exec.stem).Executor
        kw['executor_cfg'] = args.exec_config

    asig = AsyncSignal(['SIGABRT', 'SIGHUP', 'SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2', 'SIGBREAK'])
    mng = ReceiverManager(q, args.config, **kw)

    while not mng.action():
        signame = asig.wait(0.1)
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
    sys.exit(mng.exit_code)
