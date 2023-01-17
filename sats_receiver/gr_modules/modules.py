import datetime as dt
import logging
import math
import pathlib

from hashlib import sha256
from typing import Mapping, Optional, Union

import ephem
import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.filter
import gnuradio.gr

from sats_receiver import utils
from sats_receiver.gr_modules import decoders, demodulators


class RadioModule(gr.gr.hier_block2):
    def __init__(self,
                 main_tune: Union[int, float],
                 samp_rate: Union[int, float],
                 bandwidth: Union[int, float],
                 frequency: Union[int, float]):
        super(RadioModule, self).__init__(
            'Radio Module',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex)
        )

        self.enabled = False
        self.main_tune = main_tune
        self.samp_rate = samp_rate
        self.bandwidth = bandwidth
        self.frequency = frequency

        self.resamp_gcd = resamp_gcd = math.gcd(bandwidth, samp_rate)

        self.blocks_copy = gr.blocks.copy(gr.gr.sizeof_gr_complex)
        self.blocks_copy.set_enabled(self.enabled)
        self.freqshifter = gr.blocks.rotator_cc(2 * math.pi * -(main_tune - frequency) / samp_rate)
        self.resampler = gr.filter.rational_resampler_ccc(
            interpolation=bandwidth // resamp_gcd,
            decimation=samp_rate // resamp_gcd,
            taps=[],
            fractional_bw=0,
        )

        self.connect(
            self,
            self.blocks_copy,
            self.freqshifter,
            self.resampler,
            self,
        )

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.blocks_copy.set_enabled(enabled)

    def set_freq_offset(self, new_freq: Union[int, float]):
        self.freqshifter.set_phase_inc(2 * math.pi * (self.main_tune - new_freq) / self.samp_rate)


class SatRecorder(gr.gr.hier_block2):
    @staticmethod
    def _validate_config(config: Mapping) -> bool:
        return (
            all(map(lambda x: x in config,
                    [
                        # 'enabled',          # optional
                        'freq',
                        # 'freq_correction',  # optional
                        'bandwidth',
                        'mode',
                        # 'decode',           # optional

                        # 'qpsk_baudrate',    # only in QPSK demode
                        # 'qpsk_excess_bw',   # optional
                        # 'qpsk_ntaps',       # optional
                        # 'qpsk_costas_bw',   # optional
                    ]))
            and (config['mode'] != utils.Mode.QPSK or 'qpsk_baudrate' in config)
        )

    def __init__(self,
                 up: 'Satellite',
                 config: Mapping,
                 main_tune: Union[int, float],
                 samp_rate: Union[int, float]):
        f = config.get('freq')
        self.prefix = f'{self.__class__.__name__}: {up.name}: {f and f": {utils.numdisp(f)}Hz"}'

        if not self._validate_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

        super(SatRecorder, self).__init__(
            self.prefix,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.config = config
        self.radio = RadioModule(main_tune, samp_rate, self.bandwidth, self.frequency)
        self.demodulator = None
        self.post_demod = gr.blocks.float_to_complex()

        try:
            self.mode == self.decode
        except ValueError as e:
            if 'Decode' in str(e):
                x = 'decoder', self.decode
            else:
                x = 'demodulation', self.mode
            raise ValueError(f'{self.prefix}: Unknown {x[0]} `{x[1]}` for `{up.name}`')

        if self.mode == utils.Mode.AM:
            self.demodulator = gr.analog.am_demod_cf(
                channel_rate=self.bandwidth,
                audio_decim=1,
                audio_pass=5000,
                audio_stop=5500,
            )

        elif self.mode == utils.Mode.FM:
            self.demodulator = gr.analog.fm_demod_cf(
                channel_rate=self.bandwidth,
                audio_decim=1,
                deviation=self.bandwidth,
                audio_pass=16000,
                audio_stop=17000,
                gain=1.0,
                tau=0,
            )

        elif self.mode == utils.Mode.WFM:
            self.demodulator = gr.analog.wfm_rcv(
                quad_rate=self.bandwidth,
                audio_decimation=1,
            )

        elif self.mode == utils.Mode.WFM_STEREO:
            if self.bandwidth < 76800:
                raise ValueError(f'{self.prefix}: param `bandwidth` for WFM Stereo must be at least 76800, got {self.bandwidth} instead')
            self.demodulator = gr.analog.wfm_rcv_pll(
                demod_rate=self.bandwidth,
                audio_decimation=1,
                deemph_tau=50e-6,
            )
            self.connect((self.demodulator, 1), (self.post_demod, 1))

        elif self.mode == utils.Mode.QUAD:
            self.demodulator = gr.analog.quadrature_demod_cf(1)

        elif self.mode == utils.Mode.QPSK:
            self.demodulator = demodulators.QpskDemod(self.bandwidth, self.qpsk_baudrate, self.qpsk_excess_bw, self.qpsk_ntaps, self.qpsk_costas_bw)

        if self.decode == utils.Decode.APT:
            self.decoder = decoders.AptDecoder(up.name, self.bandwidth, up.output_directory, up.sat_ephem_tle, up.observer_lonlat)

        # elif self.decode == Decode.LRPT:
        #     # TODO
        #     # self.decoder =

        elif self.decode == utils.Decode.RSTREAM:
            self.decoder = decoders.RawStreamDecoder(up.name, self.bandwidth, up.output_directory)

        elif self.decode == utils.Decode.RAW:
            self.decoder = decoders.RawDecoder(up.name, self.bandwidth, up.output_directory)

        self.connect(
            self,
            self.radio,
            *((self.demodulator, self.post_demod) if self.demodulator else tuple()),
            self.decoder,
        )

    def set_freq_offset(self, new_freq: Union[int, float]):
        self.radio.set_freq_offset(new_freq)

    @property
    def is_runned(self) -> bool:
        return self.radio.enabled

    @property
    def enabled(self) -> bool:
        return self.config.get('enabled', True)

    @property
    def freq(self) -> Union[int, float]:
        return self.config['freq']

    @property
    def freq_correction(self) -> Union[int, float]:
        return self.config.get('freq_correction', 0)

    @property
    def frequency(self) -> Union[int, float]:
        return self.freq + self.freq_correction

    @property
    def bandwidth(self) -> Union[int, float]:
        return self.config['bandwidth']

    @property
    def mode(self) -> utils.Mode:
        return utils.Mode(self.config.get('mode', utils.Mode.RAW.value))

    @property
    def decode(self) -> utils.Decode:
        return utils.Decode(self.config.get('decode', utils.Decode.RAW.value)) or utils.Decode.RAW

    @property
    def qpsk_baudrate(self) -> Union[int, float]:
        return self.config['qpsk_baudrate']

    @property
    def qpsk_excess_bw(self) -> Union[int, float]:
        return self.config.get('qpsk_excess_bw')

    @property
    def qpsk_ntaps(self) -> int:
        return int(self.config.get('qpsk_ntaps'))

    @property
    def qpsk_costas_bw(self):
        return self.config.get('qpsk_costas_bw')


class Satellite(gr.gr.hier_block2):
    @staticmethod
    def _validate_config(config: Mapping) -> bool:
        return (
            all(map(lambda x: x in config,
                    [
                        'name',
                        # 'enabled',        # optional
                        # 'min_elevation',  # optional
                        'frequencies',
                        # 'doppler',          # optional
                    ]))
            and config['frequencies']
        )

    def __init__(self,
                 config: Mapping,
                 sat_ephem_tle: Optional[tuple[ephem.EarthSatellite, tuple[str, str, str]]],
                 observer_lonlat: tuple[Union[int, float], Union[int, float]],
                 main_tune: Union[int, float],
                 samp_rate: Union[int, float],
                 output_directory: pathlib.Path,
                 executor):
        n = config.get('name', '')
        self.prefix = f'{self.__class__.__name__}{n and f": {n}"}'
        self.log = logging.getLogger(self.prefix)

        if not self._validate_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

        super(Satellite, self).__init__(
            self.prefix,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.sat_ephem_tle = sat_ephem_tle
        self.observer_lonlat = observer_lonlat
        self.executor = executor
        self.config = config
        self.output_directory = output_directory / self.name
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.events: list[Optional[utils.Event]] = [None, None, None]
        self.recorders = []

        for cfg in self.frequencies:
            try:
                r = SatRecorder(self, cfg, main_tune, samp_rate)
                if r.enabled:
                    self.connect(self, r)
                    self.recorders.append(r)
            except ValueError as e:
                self.log.warning('Skip freq `%s`: %s', cfg.get('freq', 'Unknown'), e)

    @property
    def is_runned(self) -> bool:
        return any(r.is_runned for r in self.recorders)

    def start(self):
        if self.enabled and not self.is_runned:
            self.log.info('START doppler=%s mode=%s decode=%s',
                          self.doppler,
                          [r.mode.value for r in self.recorders],
                          [r.decode.value for r in self.recorders])
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self.start_event = None

            for r in self.recorders:
                r.decoder.start()
                r.radio.set_enabled(1)

    def stop(self):
        if self.is_runned:
            self.log.info('STOP')
            self.start_event = self.stop_event = None
            fin_key = sha256((self.name + str(dt.datetime.now())).encode()).hexdigest()

            for r in self.recorders:
                if r.is_runned:
                    r.radio.set_enabled(0)
                    r.decoder.finalize(self.executor, fin_key)

    def correct_doppler(self, observer: ephem.Observer):
        if self.is_runned and self.doppler:
            self.sat_ephem_tle[0].compute(observer)

            for r in self.recorders:
                if r.is_runned:
                    r.set_freq_offset(utils.doppler_shift(r.frequency, self.sat_ephem_tle[0].range_velocity))

    @property
    def name(self) -> str:
        return self.config['name']

    @property
    def enabled(self) -> bool:
        return self.config.get('enabled', True) and self.recorders

    @property
    def min_elevation(self) -> Union[int, float]:
        return self.config.get('min_elevation', 0.0)

    @property
    def frequencies(self) -> Mapping:
        return self.config['frequencies']

    @property
    def doppler(self) -> bool:
        return self.config.get('doppler', True)

    @property
    def start_event(self) -> utils.Event:
        return self.events[0]

    @start_event.setter
    def start_event(self, val: utils.Event):
        self.events[0] = val

    @property
    def stop_event(self) -> utils.Event:
        return self.events[1]

    @stop_event.setter
    def stop_event(self, val: utils.Event):
        self.events[1] = val

    @property
    def recalc_event(self) -> utils.Event:
        return self.events[2]

    @recalc_event.setter
    def recalc_event(self, val: utils.Event):
        self.events[2] = val
