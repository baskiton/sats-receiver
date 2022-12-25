import datetime as dt
import dateutil.tz
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


class SatsReceiver(gr.gr.top_block):
    def __init__(self, up, config):
        super(SatsReceiver, self).__init__('Sats Receiver', catch_exceptions=True)

        self.up = up
        self.config = {}
        self.satellites: dict[str, modules.Satellite] = {}
        self.is_runned = False

        self.signal_src = gr.blocks.null_source(gr.gr.sizeof_gr_complex)
        self.blocks_correctiq = gr.blocks.correctiq()
        self.src_null_sink = gr.blocks.null_sink(gr.gr.sizeof_gr_complex)

        if not self.update_config(config):
            raise ValueError('Receiver: %s: Invalid config!', self.name)

    def update_config(self, config, force=False):
        if force or self.config != config:
            if not self._validate_config(config):
                logging.warning('Receiver: %s: invalid new config!', self.name)
                return

            if (self.is_runned
                    and (self.source != config['source']
                         or self.serial != config.get('serial', ''))):
                logging.warning('Receiver: %s: Could not change receiver on running state!', self.name)
                return

            try:
                old_src, old_serial = self.source, self.serial
            except KeyError:
                old_src, old_serial = '', ''

            self.config = config

            if old_src != self.source or old_serial != self.serial:
                gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                'fc32', 1, '', '', [''], [''])
                logging.info('Receiver: %s: New source found', self.name)

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
                try:
                    sat = modules.Satellite(new_cfg_sats[sat_name], self.tune, self.samp_rate, self.output_directory)
                except ValueError as e:
                    logging.warning('Receiver: %s: %s: %s. Skip', self.name, sat_name, e)
                    continue

                if self.calculate_pass(sat):
                    if self.is_runned:
                        self.connect(self.blocks_correctiq, sat)
                    self.satellites[sat.name] = sat

            if to_remove_sats or to_create_sats:
                self.unlock()

            return 1

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'name',
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
        if not self.is_runned:
            logging.info('Receiver: %s: START tune=%s samp_rate=%s gain=%s biast=%s',
                         self.name, self.tune, self.samp_rate, self.gain, self.biast)

            if not LIBRTLSDR and self.source == 'rtlsdr':
                try:
                    librtlsdr.set_bt(self.biast, self.serial)
                except librtlsdr.LibRtlSdrError as e:
                    if self.biast:
                        logging.info('Receiver: %s: turn on bias-t error: %s', self.name, e)

            try:
                self.signal_src = gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                                  'fc32', 1, '', '', [''], [''])
            except RuntimeError as e:
                logging.error('Receiver: %s: cannot start: %s', self.name, e)

                self.stop()
                self.wait()

                if not LIBRTLSDR and self.source == 'rtlsdr':
                    try:
                        librtlsdr.set_bt(0, self.serial)
                    except librtlsdr.LibRtlSdrError:
                        pass

                t = self.up.now + dt.timedelta(minutes=5)
                for sat in self.satellites.values():
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
            logging.info('Receiver: %s: STOP', self.name)

            super(SatsReceiver, self).stop()
            self.is_runned = False

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
                self.disconnect(self.blocks_correctiq, sat)

    def action(self):
        if self.is_active and not self.start():
            for sat in self.satellites.values():
                if sat.is_runned and sat.doppler:
                    x = self.up.tle.get(sat.name)
                    x.compute(self.up.observer.get_obj())
                    sat.set_freq_offset(utils.doppler_shift(sat.frequency, x.range_velocity))
        else:
            x = self.is_runned

            self.stop(False)
            self.wait()

            if x and not LIBRTLSDR and self.source == 'rtlsdr':
                try:
                    librtlsdr.set_bt(0, self.serial)
                except librtlsdr.LibRtlSdrError as e:
                    logging.debug('Receiver: %s: turn off bias-t error: %s', self.name, e)

    def calculate_pass(self, sat: modules.Satellite):
        sat.events = [None, None, None]

        x = self.up.tle.get(sat.name)
        if x:
            t = self.up.now
            tt = t + dt.timedelta(hours=24)
            ltz = dateutil.tz.tzlocal()

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
                    logging.info('Receiver: %s: Sat `%s` planned on %s <-> %s', self.name, sat.name, rise_t.astimezone(ltz), set_t.astimezone(ltz))
                    break

                t = set_tt

            if t > tt:
                logging.info('Receiver: %s: Sat `%s`: No passes found for the next 24 hours', self.name, sat.name)
                self.up.scheduler.plan(tt, self.calculate_pass, sat)

            return 1

        logging.info('Receiver: %s: Sat `%s` not found in TLE. Skip', self.name, sat.name)
