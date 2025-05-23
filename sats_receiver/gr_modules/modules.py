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
from sats_receiver.observer import Observer


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

        self.freqshifter = gr.blocks.rotator_cc(2 * math.pi * (main_tune - frequency) / samp_rate)
        self.resampler = gr.filter.rational_resampler_ccc(
            interpolation=bandwidth // resamp_gcd,
            decimation=samp_rate // resamp_gcd,
            taps=[],
            fractional_bw=0,
        )

        self.connect(
            self,
            self.freqshifter,
            self.resampler,
            self,
        )

    def set_enabled(self, enabled):
        self.enabled = enabled

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

                        # 'sstv_wsr',         # optional, only in SSTV decode
                        # 'sstv_sync',        # optional, only in SSTV decode
                        # 'sstv_live_exec',   # optional, only in SSTV decode

                        # 'channels',         # only for FSK, GFSK, GMSK
                        # 'deviation_factor', # only for FSK, GFSK, GMSK, QUAD2FSK
                        # 'fsk_baudrate',     # only for FSK, GFSK, GMSK, QUAD2FSK
                        # 'proto_mode',       # only for FSK, GFSK, GMSK, QUAD2FSK

                        # 'proto_deframer',   # optional, only for PROTO
                        # 'proto_options',    # optional, only for PROTO

                        # 'grs_file',         # optional, only for SATS decode
                        # 'grs_name',         # optional, only for SATS decode
                        # 'grs_norad',        # optional, only for SATS decode
                        # 'grs_tlm_decode',   # optional, only for SATS decode

                        # 'ccc_frame_size'    # optional, only for CCSDSCC decode
                        # 'ccc_pre_deint'     # optional, only for CCSDSCC decode
                        # 'ccc_diff'          # optional, only for CCSDSCC decode
                        # 'ccc_rs_dualbasis'  # optional, only for CCSDSCC decode
                        # 'ccc_rs_interleaving'   # optional, only for CCSDSCC decode
                        # 'ccc_derandomize'   # optional, only for CCSDSCC decode

                        # 'quad_gain',        # optional, only for QUAD, QUAD2FSK demode

                        # 'raw_out_format',   # optional, only for RAW decode
                        # 'raw_out_subformat',# optional, only for RAW decode
                    ]))
            and (utils.Mode[config['mode']] != utils.Mode.QPSK or 'qpsk_baudrate' in config)
            and (
                    utils.Decode[config.get('decode')] != utils.Decode.PROTO
                    or (
                            utils.Mode[config['mode']] in (utils.Mode.FSK, utils.Mode.GFSK, utils.Mode.GMSK)
                            and 'proto_deframer' in config
                            # and 'proto_options' in config
                    )
            )
            and (utils.Mode[config['mode']] != utils.Mode.QUAD2FSK
                 or (
                         'fsk_baudrate' in config
                         and 'proto_mode' in config
                 )
            )
        )

    def __init__(self,
                 up: 'Satellite',
                 config: Mapping,
                 main_tune: Union[int, float],
                 samp_rate: Union[int, float]):
        f = config.get('freq')
        self.prefix = f'{self.__class__.__name__}: {up.name}: {f and f": {utils.num_disp(f)}Hz"}'

        if not self._validate_config(config):
            raise ValueError(f'{self.prefix}: Invalid config!')

        super(SatRecorder, self).__init__(
            self.prefix,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.satellite = up
        self.config = config
        self.radio = RadioModule(main_tune, samp_rate, self.bandwidth, self.frequency)
        self.demodulator = None
        self.post_demod = gr.blocks.float_to_complex()
        self.iq_demod = 1

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
            self.iq_demod = 0

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
            self.iq_demod = 0

        elif self.mode == utils.Mode.WFM:
            self.demodulator = gr.analog.wfm_rcv(
                quad_rate=self.bandwidth,
                audio_decimation=1,
            )
            self.iq_demod = 0

        elif self.mode == utils.Mode.WFM_STEREO:
            if self.bandwidth < 76800:
                raise ValueError(f'{self.prefix}: param `bandwidth` for WFM Stereo must be at least 76800, '
                                 f'got {self.bandwidth} instead')
            self.demodulator = gr.analog.wfm_rcv_pll(
                demod_rate=self.bandwidth,
                audio_decimation=1,
                deemph_tau=50e-6,
            )
            self.connect((self.demodulator, 1), (self.post_demod, 1))

        elif self.mode == utils.Mode.QUAD:
            self.demodulator = gr.analog.quadrature_demod_cf(self.quad_gain)
            self.iq_demod = 0

        elif self.mode == utils.Mode.SSTV_QUAD:
            self.demodulator = demodulators.SstvQuadDemod(self.bandwidth, self.demode_out_sr, self.quad_gain)
            self.iq_demod = 0

        elif self.mode in (utils.Mode.QPSK, utils.Mode.OQPSK):
            oqpsk = self.mode == utils.Mode.OQPSK
            self.demodulator = demodulators.QpskDemod(self.bandwidth, self.qpsk_baudrate, self.qpsk_excess_bw,
                                                      self.qpsk_ntaps, self.qpsk_costas_bw, oqpsk)
            self.post_demod = None

        elif self.mode in (utils.Mode.FSK, utils.Mode.GFSK, utils.Mode.GMSK):
            fsk_demod = {
                utils.Mode.FSK: demodulators.FskDemod,
                utils.Mode.GFSK: demodulators.GfskDemod,
                utils.Mode.GMSK: demodulators.GmskDemod,
            }
            self.demodulator = fsk_demod[self.mode](self.bandwidth, self.channels, self.deviation_factor)
            self.post_demod = None
            self.iq_demod = 0

        elif self.mode in (utils.Mode.USB, utils.Mode.LSB, utils.Mode.DSB):
            self.demodulator = demodulators.SsbDemod(self.bandwidth, utils.SsbMode[self.mode.name],
                                                     self.ssb_bandwidth, self.ssb_out_sr)
            self.iq_demod = 0

        elif self.mode == utils.Mode.QUAD2FSK:
            self.demodulator = demodulators.Quad2FskDemod(self.bandwidth, self.fsk_baudrate, self.deviation_factor, self.quad_gain)
            self.iq_demod = 0

        channels = getattr(self.demodulator, 'channels', (self.demode_out_sr,))
        self.decoders = []
        if self.decode == utils.Decode.APT:
            self.decoders.append(decoders.AptDecoder(self, self.bandwidth))

        # elif self.decode == Decode.LRPT:
        #     # TODO
        #     # self.decoder =

        elif self.decode == utils.Decode.CSOFT:
            self.decoders.append(decoders.ConstelSoftDecoder(self, self.bandwidth))

        elif self.decode == utils.Decode.CCSDSCC:
            self.decoders.append(decoders.CcsdsConvConcatDecoder(self, self.bandwidth))

        elif self.decode in (utils.Decode.RAW, utils.Decode.PROTO_RAW):
            _ = {
                utils.Decode.RAW: decoders.RawDecoder,
                utils.Decode.PROTO_RAW: decoders.ProtoRawDecoder,
            }
            _decoder = _[self.decode]
            if not self.iq_demod:
                self.post_demod = None

            for ch in channels:
                self.decoders.append(_decoder(self, ch, iq_in=self.iq_demod))

        elif self.decode == utils.Decode.SSTV:
            self.decoders.append(decoders.SstvDecoder(self, self.demode_out_sr))

        elif self.decode == utils.Decode.SATS:
            cfg = dict(file=self.grs_file, name=self.grs_name, norad=self.grs_norad, tlm_decode=self.grs_tlm_decode)
            self.decoders.append(decoders.SatellitesDecoder(self, self.bandwidth, cfg, 1))

        elif self.decode == utils.Decode.PROTO:
            self.decoders.append(decoders.ProtoDecoder(self, channels))
            for i in range(1, len(channels)):
                self.connect((self.demodulator, i), (self.decoders[0], i))

        x = [self, self.radio]
        if self.demodulator:
            x.append(self.demodulator)
            if self.post_demod:
                x.append(self.post_demod)

        self.connect(*x)
        for i, decoder in enumerate(self.decoders):
            self.connect((x[-1], i), decoder)

        if (self.iq_dump or self.iq_waterfall) and self.decode != utils.Decode.RAW:
            self.decoders.append(decoders.RawDecoder(self, self.bandwidth, not self.iq_dump, 1,
                                                     out_fmt=utils.RawOutFormat.NONE))
            self.connect(self.radio, self.decoders[-1])

    def start(self, observation_key: str):
        for decoder in self.decoders:
            decoder.set_observation_key(observation_key)
            decoder.start()
        self.radio.set_enabled(1)

    def stop(self):
        if self.is_runned:
            self.radio.set_enabled(0)
            for decoder in self.decoders:
                decoder.finalize()

    def set_freq_offset(self, new_freq: Union[int, float]):
        self.radio.set_freq_offset(new_freq)

    def lock_reconf(self, detach=0):
        for d in self.decoders:
            d.lock_reconf(detach)

    @property
    def is_runned(self) -> bool:
        return self.radio.enabled

    @property
    def enabled(self) -> bool:
        return self.config.get('enabled', True)

    @property
    def subname(self) -> str:
        return self.config.get('subname', '')

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
    def demode_out_sr(self) -> Union[int, float]:
        return self.config.get('demode_out_sr', self.bandwidth)

    @property
    def mode(self) -> utils.Mode:
        return utils.Mode[self.config.get('mode', utils.Mode.RAW.name)]

    @property
    def decode(self) -> utils.Decode:
        return utils.Decode[self.config.get('decode', utils.Decode.RAW.name)]

    @property
    def qpsk_baudrate(self) -> Union[int, float]:
        return self.config['qpsk_baudrate']

    @property
    def qpsk_excess_bw(self) -> Union[int, float]:
        return self.config.get('qpsk_excess_bw', 0.35)

    @property
    def qpsk_ntaps(self) -> int:
        return int(self.config.get('qpsk_ntaps', 33))

    @property
    def qpsk_costas_bw(self) -> Union[int, float]:
        return self.config.get('qpsk_costas_bw', 0.005)

    @property
    def sstv_wsr(self) -> Union[int, float]:
        return self.config.get('sstv_wsr', 16000)

    @property
    def sstv_sync(self) -> bool:
        return self.config.get('sstv_sync', True)

    @property
    def sstv_live_exec(self) -> bool:
        return self.config.get('sstv_live_exec', False)

    @property
    def channels(self) -> list[Union[int, float]]:
        return self.config.get('channels')

    @property
    def deviation_factor(self) -> Union[int, float]:
        return self.config.get('deviation_factor', 5)

    @property
    def fsk_baudrate(self) -> int:
        return self.config['fsk_baudrate']

    @property
    def proto_mode(self) -> utils.Mode:
        return utils.Mode[self.config['proto_mode']]

    @property
    def proto_deframer(self) -> utils.ProtoDeframer:
        return utils.ProtoDeframer[self.config['proto_deframer']]

    @property
    def proto_options(self) -> Mapping:
        return self.config.get('proto_options', {})

    @property
    def grs_file(self) -> Optional[pathlib.Path]:
        p = self.config.get('grs_file', None)
        if p:
            return pathlib.Path(p).expanduser()

    @property
    def grs_name(self) -> Optional[str]:
        return self.config.get('grs_name', None)

    @property
    def grs_norad(self) -> Optional[int]:
        x = self.config.get('grs_norad', None)
        if x is not None:
            return int(x)

    @property
    def grs_tlm_decode(self) -> bool:
        return self.config.get('grs_tlm_decode', True)

    @property
    def ccc_frame_size(self) -> int:
        return self.config.get('ccc_frame_size', 892)

    @property
    def ccc_pre_deint(self) -> bool:
        return self.config.get('ccc_pre_deint', False)

    @property
    def ccc_diff(self) -> bool:
        return self.config.get('ccc_diff', True)

    @property
    def ccc_rs_dualbasis(self) -> bool:
        return self.config.get('ccc_rs_dualbasis', False)

    @property
    def ccc_rs_interleaving(self) -> int:
        return self.config.get('ccc_rs_interleaving', 4)

    @property
    def ccc_derandomize(self) -> bool:
        return self.config.get('ccc_derandomize', True)

    @property
    def quad_gain(self) -> float:
        return self.config.get('quad_gain', 1.0)

    @property
    def raw_out_format(self) -> utils.RawOutFormat:
        f = self.config.get('raw_out_format', utils.RawOutFormat.WAV.name)
        return utils.RawOutFormat[f]

    @property
    def raw_out_subformat(self) -> utils.RawOutSubFormat:
        return utils.RawOutSubFormat[self.config.get('raw_out_subformat',
                                                     utils.RawOutDefaultSub[self.raw_out_format.name].value.name)]

    @property
    def ssb_bandwidth(self):
        return self.config.get('ssb_bandwidth')

    @property
    def ssb_out_sr(self):
        return self.config.get('ssb_out_sr', 8000)

    @property
    def iq_waterfall(self) -> Mapping:
        return self.config.get('iq_waterfall')

    @property
    def iq_dump(self) -> bool:
        return self.config.get('iq_dump', False)


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
                 sat_ephem_tle: tuple[ephem.EarthSatellite, tuple[str, str, str]],
                 receiver: 'SatsReceiver'):
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
        self.receiver = receiver
        self.config = config
        self.output_directory = receiver.output_directory / self.name
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.events: list[Optional[utils.Event]] = [None, None, None]
        self.recorders = []

        for cfg in self.frequencies:
            try:
                if cfg.get('enabled', True):
                    r = SatRecorder(self, cfg, receiver.tune, receiver.samp_rate)
                    self.connect(self, r)
                    self.recorders.append(r)
            except ValueError as e:
                self.log.warning('Skip freq `%s`: %s', cfg.get('freq', 'Unknown'), e)

    @property
    def observer(self) -> Observer:
        return self.receiver.up.observer

    @property
    def executor(self):
        return self.receiver.up.executor

    @property
    def is_runned(self) -> bool:
        return any(r.is_runned for r in self.recorders)

    def start(self):
        if self.enabled and not self.is_runned:
            observation_key = sha256((self.name + str(dt.datetime.now())).encode()).hexdigest()
            self.log.info('START doppler=%s mode=%s decode=%s key=%s',
                          self.doppler,
                          [r.mode.name for r in self.recorders],
                          [r.decode.name for r in self.recorders],
                          observation_key)
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self.start_event = None

            for r in self.recorders:
                r.start(observation_key)

    def stop(self):
        if self.is_runned:
            self.log.info('STOP')
            self.start_event = self.stop_event = None

            for r in self.recorders:
                if r.is_runned:
                    r.stop()

    def correct_doppler(self, observer: ephem.Observer):
        if self.is_runned and self.doppler:
            self.sat_ephem_tle[0].compute(observer)

            for r in self.recorders:
                if r.is_runned:
                    r.set_freq_offset(utils.doppler_shift(r.frequency, self.sat_ephem_tle[0].range_velocity))

    def lock_reconf(self, detach=0):
        for r in self.recorders:
            if r.is_runned:
                r.lock_reconf(detach)

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
    def frequencies(self) -> list[Mapping]:
        return self.config['frequencies']

    @property
    def doppler(self) -> bool:
        return self.config.get('doppler', True)

    @property
    def tle_strings(self) -> Optional[list[str, str]]:
        return self.config.get('tle_strings')

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


if __name__ == '__main__':
    if not SatRecorder._validate_config({
                            "enabled": True,
                            "freq": 436500000,
                            "bandwidth": 48000,
                            "mode": "RAW",
                            "decode": "RAW"
    }):
        raise ValueError
