import datetime as dt
import dateutil.tz
import enum
import logging
import pathlib

import gnuradio as gr
import gnuradio.blocks
import gnuradio.gr
import gnuradio.soapy

from sats_receiver import utils
from sats_receiver.gr_modules import modules

try:
    from sats_receiver import librtlsdr
    LIBRTLSDR = None
except OSError as e:
    LIBRTLSDR = str(e)


class RecUpdState(enum.IntEnum):
    UPD_NEED = enum.auto()
    FORCE_NEED = enum.auto()
    NO_NEED = enum.auto()


class SatsReceiver(gr.gr.top_block):
    def __init__(self, up, config):
        n = config.get('name', '')
        self.prefix = f'Receiver{n and f": {n}"}'
        self.log = logging.getLogger(self.prefix)

        super(SatsReceiver, self).__init__('Sats Receiver', catch_exceptions=True)

        self.up = up
        self.config = {}
        self.satellites: dict[str, modules.Satellite] = {}
        self.is_runned = False
        self.updated = RecUpdState.UPD_NEED

        self.signal_src = gr.blocks.null_source(gr.gr.sizeof_gr_complex)
        self.blocks_correctiq = gr.blocks.correctiq()
        self.src_null_sink = gr.blocks.null_sink(gr.gr.sizeof_gr_complex)

        if not self.update_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

    def update_config(self, config, force=False):
        if force or self.updated == RecUpdState.FORCE_NEED or self.config != config:
            if not self._validate_config(config):
                self.log.warning('invalid new config!')
                return

            if self.is_runned and config.get('enabled', True):
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
                self.signal_src.set_sample_rate(0, self.samp_rate)
                self.signal_src.set_frequency(0, self.tune)
                self.signal_src.set_frequency_correction(0, 0)
                self.signal_src.set_gain_mode(0, False)
                self.signal_src.set_gain(0, 'TUNER', self.gain)

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

            for sat_name in to_remove_sats:
                sat = self.satellites[sat_name]
                self.up.scheduler.cancel(*sat.events)
                sat.stop()
                if self.is_runned:
                    self.disconnect(self.blocks_correctiq, sat)
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
                    sat = modules.Satellite(cfg, sat_ephem_tle, self.up.observer.lonlat, self.tune, self.samp_rate, self.output_directory, self.up.executor)
                except ValueError as e:
                    self.log.warning('%s: %s. Skip', sat_name, e)
                    continue

                if self.calculate_pass(sat):
                    if self.is_runned:
                        self.connect(self.blocks_correctiq, sat)
                    self.satellites[sat.name] = sat

            if to_remove_sats or to_create_sats:
                self.unlock()

            self.updated = RecUpdState.NO_NEED

            return 1

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'name',
            # 'enabled',  # optional
            'source',
            # 'serial',   # optional
            # 'biast',    # optional
            # 'gain',     # optional
            'tune',
            'samp_rate',
            'output_directory',
            'sats',
        ]))

    @property
    def name(self):
        return self.config['name']

    @property
    def enabled(self):
        return self.config.get('enabled', True)

    @property
    def source(self):
        return self.config['source']

    @property
    def serial(self):
        return self.config.get('serial', '')

    @property
    def biast(self):
        return self.config.get('biast', False)

    @property
    def gain(self):
        return self.config.get('gain', 0)

    @property
    def tune(self):
        return self.config['tune']

    @property
    def samp_rate(self):
        return self.config['samp_rate']

    @property
    def output_directory(self):
        return pathlib.Path(self.config['output_directory']).expanduser()

    @property
    def sats(self):
        return self.config['sats']

    @property
    def is_active(self):
        return any(x.is_runned for x in self.satellites.values())

    def start(self, max_noutput_items=10000000):
        if self.enabled and not self.is_runned:
            self.log.info('START tune=%s samp_rate=%s gain=%s biast=%s',
                          self.tune, self.samp_rate, self.gain, self.biast)

            if not LIBRTLSDR and self.source == 'rtlsdr':
                try:
                    librtlsdr.set_bt(self.biast, self.serial)
                except librtlsdr.LibRtlSdrError as e:
                    if self.biast:
                        self.log.info('turn on bias-t error: %s', e)

            try:
                self.signal_src = gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                                  'fc32', 1, '', '', [''], [''])
            except RuntimeError as e:
                self.log.error('cannot start: %s', e)

                self.stop()
                self.wait()

                if not LIBRTLSDR and self.source == 'rtlsdr':
                    try:
                        librtlsdr.set_bt(0, self.serial)
                    except librtlsdr.LibRtlSdrError:
                        pass

                t = self.up.now + dt.timedelta(minutes=5)
                for sat in self.satellites.values():
                    self.up.scheduler.cancel(*sat.events)
                    if sat.enabled:
                        self.up.scheduler.plan(t, self.calculate_pass, sat)

                return 1

            self.signal_src.set_sample_rate(0, self.samp_rate)
            self.signal_src.set_frequency(0, self.tune)
            self.signal_src.set_frequency_correction(0, 0)
            self.signal_src.set_gain_mode(0, False)
            self.signal_src.set_gain(0, 'TUNER', self.gain)

            self.connect(
                self.signal_src,
                self.blocks_correctiq,
                self.src_null_sink,
            )

            for sat in self.satellites.values():
                self.connect(self.blocks_correctiq, sat)

            super(SatsReceiver, self).start(max_noutput_items)
            self.is_runned = True

    def stop(self, sched_clear=True):
        if self.is_runned:
            self.log.info('STOP')

            super(SatsReceiver, self).stop()

            self.disconnect(
                self.signal_src,
                self.blocks_correctiq,
                self.src_null_sink,
            )
            self.signal_src = gr.blocks.null_source(gr.gr.sizeof_gr_complex)

        for sat in self.satellites.values():
            if sched_clear:
                self.up.scheduler.cancel(*sat.events)
            sat.stop()
            if self.is_runned:
                self.disconnect(self.blocks_correctiq, sat)

        self.is_runned = False

    def action(self):
        if self.is_active and not self.start():
            for sat in self.satellites.values():
                sat.correct_doppler(self.up.observer.get_obj())
        else:
            x = self.is_runned

            self.stop(False)
            self.wait()

            if x and not LIBRTLSDR and self.source == 'rtlsdr':
                try:
                    librtlsdr.set_bt(0, self.serial)
                except librtlsdr.LibRtlSdrError as e:
                    self.log.debug('turn off bias-t error: %s', e)

    def calculate_pass(self, sat: modules.Satellite):
        sat.events = [None, None, None]

        x = self.up.tle.get_ephem(sat.name)
        if x:
            t = self.up.now
            tt = t + dt.timedelta(hours=24)
            ltz = dateutil.tz.tzlocal()

            self.up.scheduler.cancel(*sat.events)
            while t <= tt:
                rise_t, rise_az, culm_t, culm_alt, set_t, set_az = self.up.observer.next_pass(x, t)
                set_tt = set_t + dt.timedelta(seconds=5)
                if culm_alt >= sat.min_elevation:
                    if set_t < rise_t:
                        rise_t = t
                    sat.events = [
                        self.up.scheduler.plan(rise_t, sat.start),
                        self.up.scheduler.plan(set_t, sat.stop),
                        self.up.scheduler.plan(set_tt, self.calculate_pass, sat)
                    ]
                    self.log.info('Sat `%s` planned on %s <-> %s', sat.name, rise_t.astimezone(ltz), set_t.astimezone(ltz))
                    break

                t = set_tt

            if t > tt:
                self.log.info('Sat `%s`: No passes found for the next 24 hours', sat.name)
                self.up.scheduler.plan(tt, self.calculate_pass, sat)

            return 1

        self.log.info('Sat `%s` not found in TLE. Skip', sat.name)
