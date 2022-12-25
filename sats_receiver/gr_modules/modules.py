import logging
import math

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.filter
import gnuradio.gr

from sats_receiver import utils
from sats_receiver.gr_modules import decoders, demodulators


class RadioModule(gr.gr.hier_block2):
    def __init__(self, main_tune, samp_rate, bandwidth, frequency):
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

    def set_freq_offset(self, new_freq):
        self.freqshifter.set_phase_inc(2 * math.pi * (self.main_tune - new_freq) / self.samp_rate)


class Satellite(gr.gr.hier_block2):
    def __init__(self, config, main_tune, samp_rate, output_directory, executor):
        if not self._validate_config(config):
            raise ValueError('Observer: Invalid config!')

        self.executor = executor
        self.config = config
        self.output_directory = output_directory / self.name
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.events = [None, None, None]

        super(Satellite, self).__init__(
            'Satellite:' + self.name,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.radio = RadioModule(main_tune, samp_rate, self.bandwidth, self.frequency)
        self.demodulator = None
        self.post_demod = gr.blocks.float_to_complex()

        if self.mode == utils.Mode.AM.value:
            self.demodulator = gr.analog.am_demod_cf(
                channel_rate=self.bandwidth,
                audio_decim=1,
                audio_pass=5000,
                audio_stop=5500,
            )

        elif self.mode == utils.Mode.FM.value:
            self.demodulator = gr.analog.fm_demod_cf(
                channel_rate=self.bandwidth,
                audio_decim=1,
                deviation=self.bandwidth,
                audio_pass=16000,
                audio_stop=17000,
                gain=1.0,
                tau=0,
            )

        elif self.mode == utils.Mode.WFM.value:
            self.demodulator = gr.analog.wfm_rcv(
                quad_rate=self.bandwidth,
                audio_decimation=1,
            )

        elif self.mode == utils.Mode.WFM_STEREO.value:
            if self.bandwidth < 76800:
                raise ValueError(f'param `bandwidth` for WFM Stereo must be at least 76800, got {self.bandwidth} instead')
            self.demodulator = gr.analog.wfm_rcv_pll(
                demod_rate=self.bandwidth,
                audio_decimation=1,
                deemph_tau=50e-6,
            )
            self.connect((self.demodulator, 1), (self.post_demod, 1))

        elif self.mode == utils.Mode.QUAD.value:
            self.demodulator = gr.analog.quadrature_demod_cf(1)

        elif self.mode == utils.Mode.QPSK.value:
            self.demodulator = demodulators.QpskDemod(self.bandwidth, self.qpsk_baudrate, self.qpsk_excess_bw, self.qpsk_ntaps, self.qpsk_costas_bw)

        elif self.mode != utils.Mode.RAW.value:
            raise ValueError(f'Unknown demodulation `{self.mode}` for `{self.name}`')

        if self.decode == utils.Decode.APT.value:
            self.decoder = decoders.AptDecoder(self.bandwidth, self.output_directory)

        # elif self.decode == Decode.LRPT.value:
        #     # TODO
        #     # self.decoder =

        elif self.decode == utils.Decode.RSTREAM.value:
            self.decoder = decoders.RawStreamDecoder(self.bandwidth, self.output_directory)

        elif self.decode == utils.Decode.RAW.value:
            self.decoder = decoders.RawDecoder(self.bandwidth, self.output_directory)

        else:
            raise ValueError(f'Unknown decoder `{self.decode}` for `{self.name}`')

        self.connect(
            self,
            self.radio,
            *((self.demodulator, self.post_demod) if self.demodulator else tuple()),
            self.decoder,
        )

    def _validate_config(self, config):
        return (all(map(lambda x: x in config, [
                    'name',
                    # 'min_elevation',    # optional
                    'frequency',
                    'bandwidth',
                    'mode',
                    # 'decode',   # optional
                    # 'doppler',  # optional
                    # 'qpsk_baudrate',    # only in QPSK demode
                    # 'qpsk_excess_bw',   # optional
                    # 'qpsk_ntaps',       # optional
                    # 'qpsk_costas_bw',   # optional
                    ]))
                and (config['mode'] != utils.Mode.QPSK or 'qpsk_baudrate' in config))

    @property
    def is_runned(self):
        return self.radio.enabled

    def start(self):
        if not self.is_runned:
            logging.info('Satellite: %s: START doppler=%s mode=%s decode=%s', self.name, self.doppler, self.mode, self.decode)
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self.start_event = None
            self.decoder.start()
            self.radio.set_enabled(1)

    def stop(self):
        if self.is_runned:
            logging.info('Satellite: %s: STOP', self.name)
            self.start_event = self.stop_event = None
            self.radio.set_enabled(0)
            self.decoder.finalize(self.name, self.executor)

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
    def doppler(self):
        return self.config.get('doppler', True)

    @property
    def qpsk_baudrate(self):
        return self.config['qpsk_baudrate']

    @property
    def qpsk_excess_bw(self):
        return self.config.get('qpsk_excess_bw')

    @property
    def qpsk_ntaps(self):
        return self.config.get('qpsk_ntaps')

    @property
    def qpsk_costas_bw(self):
        return self.config.get('qpsk_costas_bw')

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
        self.radio.set_freq_offset(new_freq)
