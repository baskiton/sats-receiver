import datetime as dt
import dateutil.tz
import logging
import math
import pathlib

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.filter
import gnuradio.gr
import gnuradio.soapy


class QpskDemod(gr.gr.hier_block2):
    def __init__(self, samp_rate, baudrate, excess_bw=None, ntaps=None):
        super(QpskDemod, self).__init__(
            'QPSK Demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_float)
        )

        if excess_bw is None:
            excess_bw = 0.6
        if ntaps is None:
            ntaps = 361

        self.samp_rate = samp_rate
        self.baudrate = baudrate
        self.excess_bw = excess_bw
        self.ntaps = ntaps

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
        self.agc = gr.analog.agc_cc(0.01, 1.0, 1.0)
        self.agc.set_max_gain(10e6)
        self.costas = gr.digital.costas_loop_cc(0.004, 4, False)
        self.symbol_sync = gr.digital.symbol_sync_cc(
            detector_type=gr.digital.TED_MUELLER_AND_MULLER,
            sps=samp_rate / baudrate,
            loop_bw=(2 * math.pi) / (2 * baudrate),
            slicer=gr.digital.constellation_qpsk().base(),
        )
        self.const_decoder = gr.digital.constellation_soft_decoder_cf(gr.digital.constellation_qpsk().base())

        self.connect(
            self,
            self.agc,
            self.costas,
            self.symbol_sync,
            self.const_decoder,
            self,
        )
