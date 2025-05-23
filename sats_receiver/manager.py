import atexit
import datetime as dt
import json
import logging
import logging.handlers
import multiprocessing as mp
import pathlib
import time

from typing import Mapping

from sats_receiver import utils
from sats_receiver.executor import Executor
from sats_receiver.gr_modules.receiver import RecUpdState, SatsReceiver
from sats_receiver.observer import Observer
from sats_receiver.tle import Tle


class ReceiverManager:
    TD_ERR_DEF = dt.timedelta(seconds=5)
    TD_ACTION = dt.timedelta(seconds=1)

    def __init__(self,
                 q: mp.Queue,
                 config_filename: pathlib.Path,
                 sysu_intv=utils.SysUsage.DEFAULT_INTV,
                 executor_cls=Executor,
                 executor_cfg=None):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)
        self.exit_code = 0

        self.sysu = utils.SysUsage(self.prefix, sysu_intv)
        self.config_filename = config_filename.expanduser().absolute()
        self.config_file_stat = None
        self.config = {}

        self.receivers: dict[str, SatsReceiver] = {}
        self.stopped = False
        self.now = self.t_next = dt.datetime.now(dt.timezone.utc)
        self.file_failed_t = 0

        self.t_err = self.now
        self.td_err = self.TD_ERR_DEF

        if not self.update_config(True, True):
            raise ValueError(f'{self.prefix}: Invalid config!')

        self.observer = Observer(self.config['observer'])
        self.tle = Tle(self.config['tle'])
        self.scheduler = utils.Scheduler()
        self.executor = executor_cls(q, sysu_intv, executor_cfg)
        self.executor.start()
        atexit.register(lambda x: (x.stop(), x.join()), self.executor)

        for cfg in self.config['receivers']:
            self._add_receiver(cfg)

    def _add_receiver(self, cfg: Mapping):
        if cfg.get('enabled', True):
            try:
                rec = SatsReceiver(self, cfg)
                self.receivers[cfg['name']] = rec
                self.td_err = self.TD_ERR_DEF
                self.t_err = self.now
            except RuntimeError as e:
                if self.now >= self.t_err:
                    self.t_err = self.now + self.td_err
                    self.td_err *= 2
                    self.log.error('Skip receiver "%s": %s', cfg['name'], e)
        elif self.now >= self.t_err:
            self.t_err = self.now + self.td_err
            self.td_err *= 2
            self.log.debug('Skip disabled receiver `%s`', cfg['name'])

    @property
    def t(self) -> dt.datetime:
        """
        Set and return current time
        """

        self.now = dt.datetime.now(dt.timezone.utc)
        return self.now

    def stop(self, exit_code=0):
        for rec in self.receivers.values():
            rec.stop()

        self.executor.stop()
        self.stopped = True

        msg = 'finish'
        if exit_code:
            self.exit_code = exit_code
            msg += ' with error'
        self.log.info(msg)

    def wait(self):
        for rec in self.receivers.values():
            rec.wait()
        self.executor.join()

    def update_config(self, init=False, force=False):
        """
        :param init: True when called from __init__
        :param force: True if you need to force update
        :return: True if config update success
        """

        needs = any(i.updated != RecUpdState.NO_NEED
                    for i in self.receivers.values())

        if not (self._check_config() or force or needs):
            return

        try:
            new_cfg = json.load(self.config_filename.open())
        except (IOError, json.JSONDecodeError) as e:
            self.log.error('Error during load config: %s', e)
            return

        if not (force or needs) and new_cfg == self.config:
            return

        if not self._validate_config(new_cfg):
            self.log.warning('invalid new config!')
            return

        if new_cfg != self.config:
            self.log.debug('reconf')
            self.config = new_cfg

        if init:
            return 1

        self.observer.update_config(self.config['observer'])
        self.tle.update_config(self.config['tle'])

        configs = {cfg['name']: cfg for cfg in self.config['receivers']}
        to_add = set(configs.keys()) - set(self.receivers.keys())
        for name, receiver in self.receivers.copy().items():
            cfg = configs.get(name)
            if cfg:
                if ((force or receiver.updated == RecUpdState.FORCE_NEED)
                        or (not receiver.is_runned and receiver.updated != RecUpdState.NO_NEED)):
                    if not cfg.get('enabled', True):
                        self.log.debug('%s: stop by disabling', name)
                        receiver.stop()
                        receiver.wait()
                        self.receivers.pop(name)
                        continue

                    try:
                        receiver.update_config(cfg, force)
                    except RuntimeError as e:
                        receiver.stop()
                        receiver.wait()
                        self.receivers.pop(name)
                        self.log.error('%s: cannot update config: %s. Stop', name, e)

            else:
                receiver.stop()
                receiver.wait()
                self.receivers.pop(name)
                self.log.debug('%s: stop by deleting', name)

        for name in to_add:
            self._add_receiver(configs[name])

        return 1

    def _check_config(self):
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
                for receiver in self.receivers.values():
                    receiver.updated = RecUpdState.UPD_NEED

                return 1

    @staticmethod
    def _validate_config(config: Mapping) -> bool:
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
            now = self.t
            self.scheduler.action(now)

            if now >= self.t_next:
                self.update_config()

            # TODO: remake this. fetching weather block all system..
            need_upd = 0
            recalc = self.observer.action(now)
            if now >= self.t_next:
                need_upd = self.tle.action(now)
                self.t_next = now + self.TD_ACTION

            for receiver in self.receivers.values():
                if recalc:
                    receiver.recalculate_pass()
                if need_upd:
                    receiver.updated = RecUpdState.UPD_NEED
                receiver.action()

        except Exception as e:
            self.log.exception('%s. Exit', e)
            self.stop(1)
