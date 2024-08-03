import datetime as dt
import dateutil.tz
import enum
import logging
import pathlib

from typing import Mapping, Union

import gnuradio as gr
import gnuradio.blocks
import gnuradio.filter
import gnuradio.gr
import gnuradio.soapy

from sats_receiver.gr_modules import modules
from sats_receiver import utils


class RecUpdState(enum.IntEnum):
    UPD_NEED = enum.auto()
    FORCE_NEED = enum.auto()
    NO_NEED = enum.auto()


class SatsReceiver(gr.gr.top_block):
    def __init__(self, up, config: Mapping):
        n = config.get('name', '')
        self.prefix = f'Receiver{n and f": {n}"}'
        self.log = logging.getLogger(self.prefix)

        super(SatsReceiver, self).__init__('Sats Receiver', catch_exceptions=True)

        self.up = up
        self.config = {}
        self.satellites: dict[str, modules.Satellite] = {}
        self.is_runned = False
        self.updated = RecUpdState.UPD_NEED
        self.src_settings_keys = set()
        self.ltz = dateutil.tz.tzlocal()

        self.flow = []
        self.connector = self.signal_src = gr.blocks.null_source(gr.gr.sizeof_gr_complex)
        self.flow.append(self.signal_src)

        if self.dc_block:
            self.connector = self.blocks_correctiq = gr.blocks.correctiq()
            self.flow.append(self.blocks_correctiq)

        if self.decimation > 1:
            self.connector = self.decim = gr.filter.rational_resampler_ccc(interpolation=1, decimation=self.decimation, taps=[])
            self.flow.append(self.decim)

        self.src_null_sink = gr.blocks.null_sink(gr.gr.sizeof_gr_complex)
        self.flow.append(self.src_null_sink)

        if not self.update_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

    def update_config(self, config: Mapping, force=False):
        """
        :param config: new config
        :param force: True if you need to force update
        :return: True if config update success
        """

        if force or self.updated != RecUpdState.NO_NEED or self.config != config:
            if not self._validate_config(config):
                self.log.warning('invalid new config!')
                return

            if self.is_runned and not config.get('enabled', True):
                self.log.debug('stop by disabling')
                self.stop()
                self.wait()

            if (self.is_runned
                    and (self.source != config['source']
                         or self.serial != config.get('serial', ''))):
                self.log.warning('Could not change receiver on running state!')
                return

            try:
                old_src, old_serial = self.source, self.serial
            except KeyError:
                old_src, old_serial = '', ''

            self.config = config

            if old_src != self.source or old_serial != self.serial:
                gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                'fc32', 1, '', '', [''], [''])
                self.log.info('New source found')

            if self.is_runned:
                self.soapy_apply(0)
            else:
                self.set_biast(0)

            exist_sats = set(self.satellites.keys())
            new_cfg_sats = {i['name']: i for i in self.sats}
            new_cfg_sats_names = set(new_cfg_sats.keys())

            to_remove_sats = exist_sats - new_cfg_sats_names
            to_update_sats = exist_sats & new_cfg_sats_names
            to_create_sats = new_cfg_sats_names - exist_sats

            for sat_name in to_update_sats:
                sat = self.satellites[sat_name]
                if sat.config != new_cfg_sats:
                    to_remove_sats.add(sat_name)
                    to_create_sats.add(sat_name)

            if to_remove_sats or to_create_sats:
                self.lock()
                self.lock_reconf()

            for sat_name in to_remove_sats:
                sat = self.satellites[sat_name]
                self.up.scheduler.cancel(*sat.events)
                if sat.is_runned:
                    self.disconnect(self.connector, sat)
                sat.stop()
                del self.satellites[sat_name]

            for sat_name in to_create_sats:
                cfg = new_cfg_sats[sat_name]
                if not cfg.get('enabled', True):
                    self.log.debug('Skip disabled sat `%s`', sat_name)
                    continue

                sat_ephem_tle = self.up.tle.get(sat_name)
                if sat_ephem_tle is None:
                    self.log.info('Sat `%s` not found in TLE. Skip', sat_name)
                    continue

                try:
                    sat = modules.Satellite(cfg, sat_ephem_tle, self.up.observer, self.tune, self.samp_rate,
                                            self.output_directory, self.up.executor)
                except Exception as e:
                    self.log.warning('%s: %s. Skip', sat_name, e)
                    continue

                if self.calculate_pass(sat):
                    self.satellites[sat.name] = sat
                    if sat.is_runned:
                        self.connect(self.connector, sat)

            if to_remove_sats or to_create_sats:
                self.unlock()

            self.updated = RecUpdState.NO_NEED

            return 1

    @staticmethod
    def _validate_config(config: Mapping) -> bool:
        return all(map(lambda x: x in config, [
            'name',
            # 'enabled',  # optional
            'source',
            # 'serial',   # optional
            # 'biast',    # optional
            # 'gain',     # optional
            # 'decimation', # optional
            'tune',
            'samp_rate',
            'output_directory',
            'sats',
        ]))

    @property
    def name(self) -> str:
        return self.config['name']

    @property
    def enabled(self) -> bool:
        return self.config.get('enabled', True)

    @property
    def source(self) -> str:
        return self.config['source']

    @property
    def serial(self) -> str:
        return self.config.get('serial', '')

    @property
    def biast(self) -> bool:
        return bool(self.config.get('biast', False))

    @property
    def gain(self) -> Union[int, float]:
        """
        Receiver gain, db
        """

        return self.config.get('gain', 0)

    @property
    def tune(self) -> Union[int, float]:
        """
        Receiver tune frequency, Hz
        """

        return self.config['tune']

    @property
    def freq_correction(self) -> Union[int, float]:
        """
        Receiver frequency correction, PPM
        """

        return self.config.get('freq_correction', 0.0)

    @property
    def freq_correction_hz(self) -> int:
        """
        Receiver frequency correction, Hz
        """

        return self.config.get('freq_correction_hz', 0)

    @property
    def samp_rate(self) -> Union[int, float]:
        """
        Receiver samplerate, Hz
        """

        return self.config['samp_rate'] // self.decimation

    @property
    def src_samp_rate(self) -> Union[int, float]:
        """
        Source samplerate, Hz
        """

        return self.config['samp_rate']

    @property
    def decim_power(self) -> int:
        """
        Input stream decimation power
        """
        return self.config.get('decim_power')

    @property
    def decimation(self) -> int:
        return 1 << (self.decim_power or 0)

    @property
    def dc_block(self) -> bool:
        return self.config.get('dc_block')

    @property
    def output_directory(self) -> pathlib.Path:
        return pathlib.Path(self.config['output_directory']).expanduser()

    @property
    def sats(self) -> list[Mapping]:
        return self.config['sats']

    @property
    def is_active(self) -> bool:
        """
        True when at least one satellite recorder is running
        """

        return any(x.is_runned for x in self.satellites.values())

    def set_source(self, src):
        if self.connector == self.signal_src:
            self.connector = src
        self.signal_src = self.flow[0] = src

    def start(self, max_noutput_items=10000000):
        if self.enabled and not self.is_runned:
            self.log.info('START tune=%sHz samp_rate=%sHz gain=%s biast=%s',
                          utils.num_disp(self.tune, 3), utils.num_disp(self.samp_rate, 3),
                          self.gain, self.biast)

            for sat in self.satellites.values():
                if sat.is_runned:
                    self.disconnect(self.connector, sat)

            try:
                self.set_source(gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                                'fc32', 1, '', '', [''], ['']))
            except RuntimeError as e:
                self.log.error('cannot start: %s', e)

                self.set_biast(0, True)
                self.stop()
                self.wait()

                t = self.up.now + dt.timedelta(minutes=5)
                for sat in self.satellites.values():
                    self.up.scheduler.cancel(*sat.events)
                    if sat.enabled:
                        self.up.scheduler.plan(t, self.calculate_pass, sat)

                return 1

            self.src_settings_keys = {a.key for a in self.signal_src.get_setting_info()}
            self.soapy_apply(0)

            self.connect(*self.flow)
            for sat in self.satellites.values():
                if sat.is_runned:
                    self.connect(self.connector, sat)

            super(SatsReceiver, self).start(max_noutput_items)
            self.is_runned = True

    def stop(self, sched_clear=True):
        """
        Stop the flowgraph

        :param sched_clear: clear scheduled tasks
        """
        if self.is_runned:
            self.log.info('STOP')

            super(SatsReceiver, self).stop()

            self.disconnect(*self.flow)
            conn = self.connector
            self.set_source(gr.blocks.null_source(gr.gr.sizeof_gr_complex))

        for sat in self.satellites.values():
            if sched_clear:
                self.up.scheduler.cancel(*sat.events)
            if self.is_runned and sat.is_runned:
                try:
                    self.disconnect(conn, sat)
                except ValueError as e:
                    sat.log.warning('stop %s fail: %s', conn.name(), e)
            sat.stop()

        self.is_runned = False

    def lock_reconf(self, detach_sat=None):
        for s in self.satellites.values():
            if s.is_runned:
                s.lock_reconf(detach_sat == s)

    def sat_attach(self, sat: modules.Satellite):
        self.lock()
        self.lock_reconf()
        self.connect(self.connector, sat)
        sat.start()
        self.unlock()

    def sat_detach(self, sat: modules.Satellite):
        try:
            self.lock()
            self.lock_reconf(sat)
            if sat.is_runned:
                self.disconnect(self.connector, sat)
        except ValueError as e:
            sat.log.warning('detach %s fail: %s', self.connector.name(), e)
        finally:
            self.unlock()
            sat.stop()

    def action(self):
        if self.is_active and not self.start():
            for sat in self.satellites.values():
                sat.correct_doppler(self.up.observer.get_obj())
        elif self.is_runned:
            self.set_biast(0)
            self.stop(False)
            self.wait()

    def calculate_pass(self, sat: modules.Satellite):
        """
        Calculate satellite pass

        :return: True when calculate success
        """

        x = 0
        s = sat.tle_strings
        if s and len(s) >= 2:
            x = utils.tle_generate(sat.name, s[0], s[1], 1, self.log)
            x = x and x[0]
        if not x:
            x = self.up.tle.get_ephem(sat.name)

        if x:
            t = self.up.now
            tt = t + dt.timedelta(hours=24)

            while t <= tt:
                try:
                    rise_t, rise_az, culm_t, culm_alt, set_t, set_az = self.up.observer.next_pass(x, t)
                except ValueError as e:
                    # skip circumpolar/non-setting satellite
                    self.log.error('Sat `%s` skip: %s', e)
                    return

                set_tt = set_t + dt.timedelta(seconds=5)
                if culm_alt >= sat.min_elevation:
                    if set_t < rise_t:
                        rise_t = t
                    sat.events = [
                        None if sat.is_runned else self.up.scheduler.plan(rise_t, self.sat_attach, sat),
                        self.up.scheduler.plan(set_t, self.sat_detach, sat),
                        self.up.scheduler.plan(set_tt, self.calculate_pass, sat)
                    ]
                    self.log.info('Sat `%s` planned on %s <-> %s',
                                  sat.name, rise_t.astimezone(self.ltz), set_t.astimezone(self.ltz))
                    break

                t = set_tt

            if t > tt:
                self.log.info('Sat `%s`: No passes found for the next 24 hours', sat.name)
                self.up.scheduler.plan(tt, self.calculate_pass, sat)

            return 1

        self.log.info('Sat `%s` not found in TLE. Skip', sat.name)

    def recalculate_pass(self):
        for sat in self.satellites.values():
            self.up.scheduler.cancel(*sat.events)
            sat.events = [None, None, None]
            self.calculate_pass(sat)

    def set_biast(self, v, silent=False):
        if (not isinstance(self.signal_src, gr.blocks.null_source)
                and 'biastee' in self.src_settings_keys):
            try:
                self.signal_src.write_setting('biastee', v)
            except (ValueError, AttributeError) as e:
                if not silent:
                    self.log.warning('change bias-t error: %s', e)

    def soapy_apply(self, ch=0):
        self.signal_src.set_sample_rate(ch, self.src_samp_rate)
        self.signal_src.set_frequency(ch, self.tune + self.freq_correction_hz)
        if self.signal_src.has_frequency_correction(ch):
            self.signal_src.set_frequency_correction(ch, self.freq_correction)
        self.signal_src.set_gain_mode(ch, False)
        self.signal_src.set_gain(ch, self.gain)
        self.set_biast(self.biast)
