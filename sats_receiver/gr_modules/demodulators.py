import math

from typing import Union

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.filter
import gnuradio.gr


class QpskDemod(gr.gr.hier_block2):
    def __init__(self,
                 samp_rate: Union[int, float],
                 baudrate: Union[int, float],
                 excess_bw: Union[int, float] = None,
                 ntaps: int = None,
                 costas_bw: Union[int, float] = None):
        super(QpskDemod, self).__init__(
            'QPSK Demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_float)
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
        self.agc.set_max_gain(10e6)
        self.costas = gr.digital.costas_loop_cc(costas_bw, 4, False)
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
        self.constel_decoder = gr.digital.constellation_soft_decoder_cf(gr.digital.constellation_qpsk().base())

        self.connect(
            self,
            self.rrc,
            self.agc,
            self.costas,
            # self.symbol_sync,
            self.recov,
            self.constel_decoder,
            self,
        )


class GmskDemod(gr.gr.hier_block2):
    def __init__(self,
                 samp_rate: Union[int, float],
                 channels: list[int]):
        chan_n = len(channels)
        super(GmskDemod, self).__init__(
            'GMSK Demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_float)
            if chan_n == 1
            else gr.gr.io_signature.makev(chan_n, chan_n, [gr.gr.sizeof_float] * chan_n)
        )

        self.samp_rate = samp_rate
        self._channels = channels
        self.wsr = int(max(channels) * 2)
        self.resamp_gcd = math.gcd(samp_rate, self.wsr)
        self._chans = []

        self.resamp = gr.filter.rational_resampler_ccc(
            interpolation=(self.wsr // self.resamp_gcd),
            decimation=(samp_rate // self.resamp_gcd),
            taps=[],
            fractional_bw=0,
        )

        self.connect(self, self.resamp)

        for i, rate in enumerate(channels):
            gmsk_demod = gr.digital.gmsk_demod(
                samples_per_symbol=self.wsr // rate,
                gain_mu=0.175,
                mu=0.5,
                omega_relative_limit=0.005,
                freq_error=0.0,
                verbose=False,
                log=False
            )
            uchtf = gr.blocks.uchar_to_float()
            self.connect(
                self.resamp,
                gmsk_demod,
                uchtf,
                (self, i),
            )
            self._chans.append((gmsk_demod, uchtf))

    @property
    def channels(self):
        return self._channels
