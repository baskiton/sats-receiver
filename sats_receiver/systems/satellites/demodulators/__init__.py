import math

import gnuradio as gr
import gnuradio.analog
import gnuradio.digital
import gnuradio.gr

from satellites.utils.options_block import options_block
from satellites.components.demodulators import fsk_demodulator


class GfskDemod(gr.gr.hier_block2, options_block):
    _DEF_GAIN_MU = 0.175
    _DEF_OMEGA_RELATIVE_LIMIT = 0.005
    _DEF_FREQ_ERROR = 0.0
    _DEF_DEVIATION_HZ = 5000

    def __init__(self, baudrate, samp_rate, iq, deviation=None,
                 subaudio=False, dc_block=True, dump_path=None,
                 options=None):
        gr.gr.hier_block2.__init__(
            self,
            'gfsk_demodulator',
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex if iq else gr.gr.sizeof_float),
            gr.gr.io_signature(1, 1, gr.gr.sizeof_float))
        options_block.__init__(self, options)

        sps = samp_rate / baudrate
        _omega_relative_limit = self.options.omega_relative_limit
        _freq_error = self.options.freq_error
        _gain_mu = self.options.gain_mu
        _deviation = self.options.deviation
        _sensitivity = 2 * math.pi * _deviation

        _omega = sps * (1 + _freq_error)
        _gain_omega = 0.25 * _gain_mu * _gain_mu
        _ted_gain = 1.0
        _damping = 1.0
        # critically damped
        _loop_bw = -math.log((_gain_mu + _gain_omega) / (-2.0) + 1)
        _max_dev = _omega_relative_limit * sps

        if iq:
            self.fmdemod = gr.analog.quadrature_demod_cf(1.0 / _sensitivity)
            self.connect(self, self.fmdemod)
        else:
            self.fmdemod = self

        self.clock_recovery = gr.digital.symbol_sync_ff(
            gr.digital.TED_MUELLER_AND_MULLER,
            _omega,
            _loop_bw,
            _damping,
            _ted_gain,  # Expected TED gain
            _max_dev,
            1,  # Output sps
            gr.digital.constellation_bpsk().base(),
            gr.digital.IR_MMSE_8TAP,
            128,
            [],
        )

        self.connect(self.fmdemod, self.clock_recovery, self)

    @classmethod
    def add_options(cls, parser):
        parser.add_argument('--gain-mu', type=float, default=cls._DEF_GAIN_MU,
                            help='Symbol Sync M&M gain mu [default=%default] (GFSK/PSK)')
        parser.add_argument('--omega-relative-limit', type=float, default=cls._DEF_OMEGA_RELATIVE_LIMIT,
                            help='Symbol Sync M&M omega relative limit [default=%default] (GFSK/PSK)')
        parser.add_argument('--freq-error', type=float, default=cls._DEF_FREQ_ERROR,
                            help='Symbol Sync M&M frequency error [default=%default] (GFSK)')
        parser.add_argument('--deviation', type=float, default=cls._DEF_DEVIATION_HZ,
                            help='Deviation (Hz) [default=%(default)r]')


class GmskDemod(fsk_demodulator):
    _DEF_DEMOD_GAIN = math.pi * 4

    def __init__(self, baudrate, samp_rate, iq, deviation=None,
                 subaudio=False, dc_block=True, dump_path=None,
                 options=None):
        super().__init__(baudrate, samp_rate, iq, deviation, subaudio, dc_block, dump_path, options)
        if iq:
            self.demod.set_gain(self.options.demod_gain)

    @classmethod
    def add_options(cls, parser):
        super().add_options(parser)
        parser.add_argument('--demod_gain', type=float, default=cls._DEF_DEMOD_GAIN,
                            help='Quadrature/FM demodulator gain [default=%default] (GMSK)')
