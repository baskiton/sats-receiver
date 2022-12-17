import datetime as dt
import dateutil.tz
import logging
import math
import pathlib

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.filter
import gnuradio.gr
import gnuradio.soapy

from sats_receiver import utils
from sats_receiver.gr_modules import decoders


class RadioModule(gr.gr.hier_block2):
    def __init__(self, up, main_tune, samp_rate):
        super(RadioModule, self).__init__(
            'Radio Module',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex)
        )

        self.up = up
        self.enabled = False
        self.main_tune = main_tune
        self.samp_rate = samp_rate

        self.resamp_gcd = resamp_gcd = math.gcd(up.bandwidth, samp_rate)

        self.blocks_copy = gr.blocks.copy(gr.gr.sizeof_gr_complex)
        self.blocks_copy.set_enabled(self.enabled)
        self.freqshifter = gr.blocks.rotator_cc(2 * math.pi * -(main_tune - up.frequency) / samp_rate)
        self.resampler = gr.filter.rational_resampler_ccc(
            interpolation=up.bandwidth // resamp_gcd,
            decimation=samp_rate // resamp_gcd,
            taps=[],
            fractional_bw=0,
        )
        self.demodulator = None
        self.float_to_complex = gr.blocks.float_to_complex()

        if up.mode == utils.Mode.AM.value:
            self.demodulator = gr.analog.am_demod_cf(
                channel_rate=up.bandwidth,
                audio_decim=1,
                audio_pass=5000,
                audio_stop=5500,
            )
        elif up.mode == utils.Mode.FM.value:
            self.demodulator = gr.analog.fm_demod_cf(
                channel_rate=up.bandwidth,
                audio_decim=1,
                deviation=up.bandwidth,
                audio_pass=16000,
                audio_stop=17000,
                gain=1.0,
                tau=0,
            )
        elif up.mode == utils.Mode.WFM.value:
            self.demodulator = gr.analog.wfm_rcv(
                quad_rate=up.bandwidth,
                audio_decimation=1,
            )
        elif up.mode == utils.Mode.WFM_STEREO.value:
            if up.bandwidth < 76800:
                raise ValueError(f'param `bandwidth` for WFM Stereo must be at least 76800, got {up.bandwidth} instead')
            self.demodulator = gr.analog.wfm_rcv_pll(
                demod_rate=up.bandwidth,
                audio_decimation=1,
                deemph_tau=50e-6,
            )
            self.connect((self.demodulator, 1), (self.float_to_complex, 1))
        elif up.mode == utils.Mode.QUAD.value:
            self.demodulator = gr.analog.quadrature_demod_cf(1)
        elif up.mode != utils.Mode.RAW.value:
            raise ValueError(f'Unknown demodulation `{up.mode}` for `{up.name}`')

        self.connect(
            self,
            self.blocks_copy,
            self.freqshifter,
            self.resampler,
            *((self.demodulator, self.float_to_complex) if self.demodulator else tuple()),
            self,
        )

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.blocks_copy.set_enabled(enabled)

    def set_freq_offset(self, new_freq):
        self.freqshifter.set_phase_inc(2 * math.pi * (self.main_tune - new_freq) / self.samp_rate)


class Satellite(gr.gr.hier_block2):
    def __init__(self, config, main_tune, samp_rate):
        if not self._validate_config(config):
            raise ValueError('Observer: Invalid config!')

        self.config = config
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.events = [None, None, None]

        super(Satellite, self).__init__(
            'Satellite:' + self.name,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.radio = RadioModule(self, main_tune, samp_rate)

        if self.decode == utils.Decode.APT.value:
            self.decoder = decoders.AptDecoder(self.bandwidth, self.output_directory)
        # elif self.decode == Decode.LRPT.value:
        #     # TODO
        #     # self.decoder =
        elif self.decode == utils.Decode.RAW.value:
            self.decoder = decoders.RawDecoder(self.bandwidth, self.output_directory)
        else:
            raise ValueError(f'Unknown decoder `{self.decode}` for `{self.name}`')

        self.connect(self, self.radio, self.decoder)

    def _validate_config(self, config):
        return all(map(lambda x: x in config, [
            'name',
            # 'min_elevation',    # optional
            'frequency',
            'bandwidth',
            'mode',
            # 'decode',   # optional
            'output_directory',
        ]))

    @property
    def is_runned(self):
        return self.radio.enabled

    def start(self):
        if not self.is_runned:
            logging.debug('Satellite %s: start', self.name)
            self.start_event = None
            self.decoder.start()
            self.radio.set_enabled(1)

    def stop(self):
        if self.is_runned:
            logging.debug('Satellite %s: stop', self.name)
            self.start_event = self.stop_event = None
            self.radio.set_enabled(0)
            self.decoder.finalize()

    @property
    def name(self):
        return self.config['name']

    @property
    def min_elevation(self):
        return self.config.get('min_elevation', 0.0)

    @property
    def frequency(self):
        return self.config['frequency']

    @property
    def bandwidth(self):
        return self.config['bandwidth']

    @property
    def mode(self):
        return self.config['mode']

    @property
    def decode(self):
        return self.config.get('decode', 'RAW')

    @property
    def output_directory(self):
        return pathlib.Path(self.config['output_directory']).expanduser()

    @property
    def start_event(self):
        return self.events[0]

    @start_event.setter
    def start_event(self, val):
        self.events[0] = val

    @property
    def stop_event(self):
        return self.events[1]

    @stop_event.setter
    def stop_event(self, val):
        self.events[1] = val

    @property
    def recalc_event(self):
        return self.events[2]

    @recalc_event.setter
    def recalc_event(self, val):
        self.events[2] = val

    def set_freq_offset(self, new_freq):
        logging.debug('Satellite %s: freq to %s', self.name, new_freq)
        self.radio.set_freq_offset(new_freq)


class SatsReceiver(gr.gr.top_block):
    def __init__(self, up, config):
        super(SatsReceiver, self).__init__('Sats Receiver', catch_exceptions=True)

        self.up = up
        self.config = {}
        self.satellites: dict[str, Satellite] = {}
        self.is_runned = False

        self.signal_src = gr.blocks.null_source(gr.gr.sizeof_gr_complex)
        self.blocks_correctiq = gr.blocks.correctiq()
        self.src_null_sink = gr.blocks.null_sink(gr.gr.sizeof_gr_complex)

        if not self.update_config(config):
            raise ValueError('SatsReceiver: Invalid config!')

    def update_config(self, config, force=False):
        if force or self.config != config:
            if not self._validate_config(config):
                logging.warning('Observer: invalid new config!')
                return

            try:
                old_src, old_serial = self.source, self.serial
            except KeyError:
                old_src, old_serial = '', ''

            self.config = config

            if old_src != self.source or old_serial != self.serial:
                if self.is_runned:
                    self.lock()
                    self.disconnect(self.signal_src, self.blocks_correctiq)

                new_src = gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                          'fc32', 1, '', '', [''], [''])
                logging.info('SatsReceiver: New source found')

                if self.is_runned:
                    self.signal_src = new_src
                    self.connect(self.signal_src, self.blocks_correctiq)
                    self.unlock()
                else:
                    del new_src

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
                    sat = Satellite(new_cfg_sats[sat_name], self.tune, self.samp_rate)
                except ValueError as e:
                    logging.warning('Receiver: %s: %s. Skip', self.name, e)
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
            # 'gain',     # optional
            'tune',
            'samp_rate',
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
    def gain(self):
        return self.config.get('gain', 0)

    @property
    def tune(self):
        return self.config['tune']

    @property
    def samp_rate(self):
        return self.config['samp_rate']

    @property
    def sats(self):
        return self.config['sats']

    @property
    def is_active(self):
        return any(x.is_runned for x in self.satellites.values())

    def start(self, max_noutput_items=10000000):
        if not self.is_runned:
            logging.info('Receiver: %s: start', self.name)

            self.signal_src = gr.soapy.source(f'driver={self.source}{self.serial and f",serial={self.serial}"}',
                                              'fc32', 1, '', '', [''], [''])
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
            logging.info('Receiver: %s: stop', self.name)

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
        if self.is_active:
            self.start()
            for sat in self.satellites.values():
                if sat.is_runned:
                    x = self.up.tle.get(sat.name)
                    x.compute(self.up.observer.get_obj())
                    sat.set_freq_offset(utils.doppler_shift(sat.frequency, x.range_velocity))
        else:
            self.stop(False)

    def calculate_pass(self, sat: Satellite):
        sat.events = [None, None, None]

        x = self.up.tle.get(sat.name)
        if x:
            t = self.up.now
            tt = t + dt.timedelta(hours=24)
            ltz = dateutil.tz.tzlocal()

            while t <= tt:
                rise_t, rise_az, culm_t, culm_alt, set_t, set_az = self.up.observer.next_pass(x, t)
                if culm_alt >= sat.min_elevation:
                    logging.info('Receiver: Sat `%s` planned on %s <-> %s', sat.name, rise_t.astimezone(ltz), set_t.astimezone(ltz))
                    sat.events = [
                        self.up.scheduler.plan(rise_t, sat.start),
                        self.up.scheduler.plan(set_t, sat.stop),
                        self.up.scheduler.plan(set_t, self.calculate_pass, sat, prior=1)
                    ]
                    break

                t = set_t

            if t > tt:
                logging.info('Receiver: Sat `%s`: No passes found for the next 24 hours', sat.name)
                self.up.scheduler.plan(tt, self.calculate_pass, sat)

            return 1

        logging.info('Receiver: Sat `%s` not found in TLE. Skip', sat.name)
