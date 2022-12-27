import atexit
import datetime as dt
import json
import logging
import multiprocessing as mp
import pathlib
import time
import queue

from sats_receiver.gr_modules.receiver import SatsReceiver
from sats_receiver.observer import Observer
from sats_receiver.tle import Tle
from sats_receiver.utils import Scheduler, SysUsage

# Rate Ranges:
#     137...138 MHz: 1.024
#     144...146 MHz: 2.16
#     435...438 MHz: 3.2


class Executor(mp.Process):
    def __init__(self, sysu_intv=SysUsage.DEFAULT_INTV):
        super().__init__(daemon=False)

        self.sysu_intv = sysu_intv
        self.q = mp.Queue()

    def run(self):
        logging.debug('Executor: start')

        sysu = SysUsage('Executor', self.sysu_intv)

        while 1:
            sysu.collect()

            try:
                x = self.q.get(timeout=1)
            except queue.Empty:
                continue

            if x == '.':
                break

            try:
                fn, args, kwargs = x
            except ValueError:
                logging.error('Executor: invalid task: %s', x)
                continue

            if callable(fn):
                try:
                    fn(*args, **kwargs)
                except Exception:
                    logging.exception('Executor: %s with args=%s kwargs=%s', fn, args, kwargs)

        logging.debug('Executor: finish')

    def execute(self, fn, *args, **kwargs):
        self.q.put((fn, args, kwargs))

    def stop(self):
        self.q.put('.')


class ReceiverManager:
    def __init__(self, config_filename: pathlib.Path, sysu_intv=SysUsage.DEFAULT_INTV):
        self.sysu = SysUsage('ReceiverManager', sysu_intv)
        self.config_filename = config_filename
        self.config_file_stat = None
        self.config = {}

        self.receivers: dict[str, SatsReceiver] = {}
        self.stopped = False
        self.now = dt.datetime.now(dt.timezone.utc)
        self.file_failed_t = 0

        if not self.update_config(True, True):
            raise ValueError('ReceiverManager: Invalid config!')

        self.observer = Observer(self.config['observer'])
        self.tle = Tle(self.config['tle'])
        self.scheduler = Scheduler()
        self.executor = Executor(sysu_intv)
        self.executor.start()
        atexit.register(lambda x: (x.stop(), x.join()), self.executor)

        for cfg in self.config['receivers']:
            self._add_receiver(cfg)

    def _add_receiver(self, cfg):
        if cfg.get('enabled', True):
            try:
                rec = SatsReceiver(self, cfg)
                self.receivers[cfg['name']] = rec
            except RuntimeError as e:
                logging.error('ReceiverManager: Skip receiver "%s": %s', cfg['name'], e)
        else:
            logging.debug('ReceiverManager: Skip disabled receiver `%s`', cfg['name'])

    @property
    def t(self):
        self.now = dt.datetime.now(dt.timezone.utc)
        return self.now

    def stop(self):
        for rec in self.receivers.values():
            rec.stop()

        self.stopped = True

        logging.info('ReceiverManager: finish')

    def wait(self):
        for rec in self.receivers.values():
            rec.wait()

    def update_config(self, init=False, force=False):
        if not self._check_config():
            return

        try:
            new_cfg = json.load(self.config_filename.open())
        except (IOError, json.JSONDecodeError) as e:
            logging.error('ReceiverManager: Error during load config: %s', e)
            return

        if new_cfg == self.config:
            return

        if not self._validate_config(new_cfg):
            logging.warning('ReceiverManager: invalid new config!')
            return

        logging.debug('ReceiverManager: reconf')
        self.config = new_cfg

        if init:
            return 1

        self.observer.update_config(new_cfg['observer'])
        self.tle.update_config(new_cfg['tle'])

        for cfg in new_cfg['receivers']:
            x = self.receivers.get(cfg['name'])
            if x:
                if not cfg.get('enabled', True):
                    logging.debug('ReceiverManager: %s: stop by disabling', x.name)
                    x.stop()
                    x.wait()
                    self.receivers.pop(x.name)
                    continue

                try:
                    x.update_config(cfg, force)
                except RuntimeError as e:
                    self.receivers.pop(cfg['name'])
                    logging.error('ReceiverManager: Delete receiver "%s" with new config: %s',
                                  cfg['name'], e)
            else:
                self._add_receiver(cfg)

        return 1

    def _check_config(self):
        try:
            st = self.config_filename.stat()
        except FileNotFoundError:
            t = time.monotonic()
            if t - self.file_failed_t > 300:
                logging.error('ReceiverManager: Config file does\'t exist: %s', self.config_filename)
                self.file_failed_t = t
            return

        old = self.config_file_stat
        if not old or (st.st_ino != old.st_ino
                       or st.st_dev != old.st_dev
                       or st.st_mtime != old.st_mtime
                       or st.st_size != old.st_size
                       or st.st_mode != old.st_mode):
            self.config_file_stat = st
            return 1

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'observer',
            'tle',
            'receivers',
        ]))

    @property
    def recs_runned(self):
        return any(x.is_runned for x in self.receivers.values())

    def action(self):
        self.sysu.collect()

        if self.stopped:
            return 1

        try:
            self.update_config()
            self.scheduler.action()
            self.observer.action(self.t)
            if not self.recs_runned and self.tle.action(self.now):
                for cfg in self.config['receivers']:
                    x = self.receivers.get(cfg['name'])
                    if x:
                        try:
                            x.update_config(cfg, True)
                        except RuntimeError as e:
                            logging.error('ReceiverManager: %s: cannot update config: %s. Stop', x.name, e)
                            x.stop()
                            x.wait()

            for rn, r in self.receivers.items():
                r.action()
        except Exception as e:
            logging.exception('ReceiverManager: %s. Exit', e)
            self.stop()
