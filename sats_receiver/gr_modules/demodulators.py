import logging
import math

from typing import Union, List

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.fft
import gnuradio.filter
import gnuradio.gr
import satellites.components.demodulators as grs_demodulators

from sats_receiver.gr_modules.epb import DelayOneImag


class QpskDemod(gr.gr.hier_block2):
    def __init__(self,
                 samp_rate: Union[int, float],
                 baudrate: Union[int, float],
                 excess_bw: Union[int, float] = None,
                 ntaps: int = None,
                 costas_bw: Union[int, float] = None,
                 oqpsk=False):
        super(QpskDemod, self).__init__(
            'QPSK Demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex)
        )

        if excess_bw is None:
            excess_bw = 0.35
        if ntaps is None:
            ntaps = 33
        if costas_bw is None:
            costas_bw = 0.005

        self.samp_rate = samp_rate
        self.baudrate = baudrate
        self.excess_bw = excess_bw
        self.ntaps = int(ntaps)

        self.rrc = gr.filter.fir_filter_ccf(
            1,
            gr.filter.firdes.root_raised_cosine(
                gain=1,
                sampling_freq=samp_rate,
                symbol_rate=baudrate,
                alpha=excess_bw,
                ntaps=ntaps
            )
        )
        self.agc = gr.analog.agc_cc(0.1, 1.0, 1.0)
        self.agc.set_max_gain(65536)
        self.costas = gr.digital.costas_loop_cc(costas_bw, 4, False)

        if oqpsk:
            self.after_costas = DelayOneImag()
            self.connect(self.costas, self.after_costas)
        else:
            self.after_costas = self.costas
        # self.symbol_sync = gr.digital.symbol_sync_cc(
        #     detector_type=gr.digital.TED_MUELLER_AND_MULLER,
        #     sps=samp_rate / baudrate,
        #     loop_bw=(2 * math.pi) / (2 * baudrate),
        #     damping_factor=1.0,
        #     ted_gain=1.0,
        #     max_deviation=1.5,
        #     osps=1,
        #     slicer=gr.digital.constellation_qpsk().base(),
        #     interp_type=gr.digital.IR_MMSE_8TAP,
        #     n_filters=128,
        #     taps=[],
        # )
        self.recov = gr.digital.clock_recovery_mm_cc(
            omega=samp_rate / baudrate,
            gain_omega=1e-6,
            mu=0.5,
            gain_mu=0.01,
            omega_relative_limit=0.01,
        )

        self.connect(
            self,
            self.rrc,
            self.agc,
            self.costas,
        )
        self.connect(
            self.after_costas,
            # self.symbol_sync,
            self.recov,
            self,
        )


class SstvQuadDemod(gr.gr.hier_block2):
    def __init__(self,
                 samp_rate: Union[int, float],
                 out_rate: Union[int, float],
                 quad_gain: Union[int, float] = 1.0):
        super(SstvQuadDemod, self).__init__(
            'SSTV Quad Demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_float)
        )

        self.out_rate = out_rate
        dev_factor = 8
        deviation_hz = out_rate / dev_factor
        carson_cutoff = abs(deviation_hz) + out_rate / 2
        resamp_gcd = math.gcd(samp_rate, out_rate)

        self.lpf = gr.filter.fir_filter_ccf(
            1,
            gr.filter.firdes.low_pass(
                gain=1,
                sampling_freq=samp_rate,
                cutoff_freq=carson_cutoff,
                transition_width=(carson_cutoff * 0.1),
                window=gr.fft.window.WIN_HAMMING,
                param=6.76,
            )
        )
        self.resamp = gr.filter.rational_resampler_ccc(
            interpolation=(out_rate // resamp_gcd),
            decimation=(samp_rate // resamp_gcd),
            taps=[],
            fractional_bw=0,
        )
        self.quad = gr.analog.quadrature_demod_cf(quad_gain)

        self.connect(self, self.lpf, self.resamp, self.quad, self)


class FskDemod(gr.gr.hier_block2):
    def __init__(self,
                 samp_rate: Union[int, float],
                 channels: List[Union[int, float]] = None,
                 deviation_factor: Union[int, float] = 5):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        if channels is None:
            channels = [1200, 2400, 4800, 9600]
        if not channels:
            raise ValueError('No channels')

        for baud in channels.copy():
            if baud >= samp_rate:
                self.log.warning('Invalid baud %s. Skip', baud)
                channels.remove(baud)

        self._channels = channels
        chans = len(channels)
        super(FskDemod, self).__init__(
            self.prefix.partition('Demod')[0] + ' Demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_float)
            if chans == 1
            else gr.gr.io_signature.makev(chans, chans, [gr.gr.sizeof_float] * chans)
        )

        self.channel_demod = {}
        for i, baud in enumerate(channels):
            deviation = baud / deviation_factor
            demod = grs_demodulators.fsk_demodulator(baud, samp_rate, 1, deviation, dc_block=0)
            self.channel_demod[baud] = demod
            self.connect(self, demod, (self, i))

    @property
    def channels(self):
        return self._channels


class GfskDemod(FskDemod):
    pass


class GmskDemod(FskDemod):
    pass
