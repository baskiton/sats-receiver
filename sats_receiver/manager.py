import atexit
import datetime as dt
import json
import logging
import multiprocessing as mp
import pathlib
import time
import queue

from sats_receiver.gr_modules.receiver import RecUpdState, SatsReceiver
from sats_receiver.observer import Observer
from sats_receiver.tle import Tle
from sats_receiver.utils import Scheduler, SysUsage

# Rate Ranges:
#     137...138 MHz: 1.024
#     144...146 MHz: 2.16
#     435...438 MHz: 3.2


class Executor(mp.Process):
    def __init__(self, sysu_intv=SysUsage.DEFAULT_INTV):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        super().__init__(daemon=False)

        self.sysu_intv = sysu_intv
        self.q = mp.Queue()

    def run(self):
        self.log.debug('start')

        sysu = SysUsage(self.prefix, self.sysu_intv)

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
                self.log.error('invalid task: %s', x)
                continue

            if callable(fn):
                try:
                    fn(*args, **kwargs)
                except Exception:
                    self.log.exception('%s with args=%s kwargs=%s', fn, args, kwargs)

        self.log.debug('finish')

    def execute(self, fn, *args, **kwargs):
        self.q.put((fn, args, kwargs))

    def stop(self):
        self.q.put('.')


class ReceiverManager:
    def __init__(self, config_filename: pathlib.Path, sysu_intv=SysUsage.DEFAULT_INTV):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        self.sysu = SysUsage(self.prefix, sysu_intv)
        self.config_filename = config_filename
        self.config_file_stat = None
        self.config = {}

        self.receivers: dict[str, SatsReceiver] = {}
        self.stopped = False
        self.now = dt.datetime.now(dt.timezone.utc)
        self.file_failed_t = 0

        if not self.update_config(True, True):
            raise ValueError(f'{self.prefix}: Invalid config!')

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
                self.log.error('Skip receiver "%s": %s', cfg['name'], e)
        else:
            self.log.debug('Skip disabled receiver `%s`', cfg['name'])

    @property
    def t(self):
        self.now = dt.datetime.now(dt.timezone.utc)
        return self.now

    def stop(self):
        for rec in self.receivers.values():
            rec.stop()

        self.executor.stop()
        self.stopped = True

        self.log.info('finish')

    def wait(self):
        for rec in self.receivers.values():
            rec.wait()
        self.executor.join()

    def update_config(self, init=False, force=False):
        if not self._check_config():
            return

        try:
            new_cfg = json.load(self.config_filename.open())
        except (IOError, json.JSONDecodeError) as e:
            self.log.error('Error during load config: %s', e)
            return

        if new_cfg == self.config:
            return

        if not self._validate_config(new_cfg):
            self.log.warning('invalid new config!')
            return

        self.log.debug('reconf')
        self.config = new_cfg

        if init:
            return 1

        self.observer.update_config(new_cfg['observer'])
        self.tle.update_config(new_cfg['tle'])

        for cfg in new_cfg['receivers']:
            x = self.receivers.get(cfg['name'])
            if x:
                if ((force or x.updated == RecUpdState.FORCE_NEED)
                        or (not x.is_runned and x.updated != RecUpdState.NO_NEED)):
                    if not cfg.get('enabled', True):
                        self.log.debug('%s: stop by disabling', x.name)
                        x.stop()
                        x.wait()
                        self.receivers.pop(x.name)
                        continue

                    try:
                        x.update_config(cfg, force)
                    except RuntimeError as e:
                        x.stop()
                        x.wait()
                        self.receivers.pop(x.name)
                        self.log.error('%s: cannot update config: %s. Stop', x.name, e)

            else:
                self._add_receiver(cfg)

        return 1

    def _check_config(self):
        x = any(i.updated == RecUpdState.FORCE_NEED
                for i in self.receivers.values())

        try:
            st = self.config_filename.stat()
        except FileNotFoundError:
            t = time.monotonic()
            if t - self.file_failed_t > 300:
                self.log.error('Config file does\'t exist: %s', self.config_filename)
                self.file_failed_t = t
        else:
            old = self.config_file_stat
            if not old or (st.st_ino != old.st_ino
                           or st.st_dev != old.st_dev
                           or st.st_mtime != old.st_mtime
                           or st.st_size != old.st_size
                           or st.st_mode != old.st_mode):
                self.config_file_stat = st
                x = 1
                for i in self.receivers.values():
                    i.updated = RecUpdState.UPD_NEED

        return x

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'observer',
            'tle',
            'receivers',
        ]))

    def action(self):
        self.sysu.collect()

        if self.stopped:
            return 1

        try:
            self.update_config()
            self.scheduler.action()
            x = bool(self.observer.action(self.t))
            x += bool(self.tle.action(self.now))
            if x:
                for i in self.receivers.values():
                    i.updated = RecUpdState.FORCE_NEED

            for rn, r in self.receivers.items():
                r.action()
        except Exception as e:
            self.log.exception('%s. Exit', e)
            self.stop()
