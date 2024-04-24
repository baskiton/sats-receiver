import logging
import logging.handlers
import pathlib
import tempfile
import time

from test import support
from unittest import TestCase

import gnuradio as gr
import gnuradio.blocks
import gnuradio.gr
import numpy as np

from sats_receiver.gr_modules.epb.prober import Prober


HERE = pathlib.Path(__file__).parent
FILES = HERE / 'files'
TIMEOUT = support.SHORT_TIMEOUT


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class DemodTopBlock(gr.gr.top_block):
    def __init__(self,
                 fp: pathlib.Path,
                 out_dir: pathlib.Path,
                 samp_rate: int,
                 channels: list[int],
                 demodulator):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        super(DemodTopBlock, self).__init__('DemodTopBlock', catch_exceptions=False)

        self.samp_rate = samp_rate
        self.channels = channels

        self.fsrc = gr.blocks.file_source(gr.gr.sizeof_gr_complex, str(fp), False, 0, samp_rate)
        self.thr = gr.blocks.throttle(gr.gr.sizeof_gr_complex, samp_rate, True)
        self.ctr = gr.blocks.complex_to_real()
        self.probe = Prober()
        self.demod = demodulator
        self.fsinks = {}
        for rate in channels:
            fn = str(out_dir / str(rate))
            fsink = gr.blocks.file_sink(gr.gr.sizeof_float, fn, False)
            fsink.set_unbuffered(False)
            self.fsinks[rate] = fn, fsink

        self.connect(
            self.fsrc,
            self.thr,
            self.demod,
        )
        self.connect(self.thr, self.ctr, self.probe)
        for i, rate in enumerate(channels):
            self.connect((self.demod, i), self.fsinks[rate][1])

    def start(self, max_noutput_items=10000000):
        self.log.info('START')
        super(DemodTopBlock, self).start(max_noutput_items)

    def stop(self):
        self.log.info('STOP')
        super(DemodTopBlock, self).stop()
        for fn, fsink in self.fsinks.values():
            fsink.do_update()
            fsink.close()

    def wait(self):
        super(DemodTopBlock, self).wait()


# class TestDemod(TestCase):
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.out_dir = tempfile.TemporaryDirectory('.d', 'sats-receiver-test', ignore_cleanup_errors=True)
#         cls.out_dp = pathlib.Path(cls.out_dir.name)
#
#     @classmethod
#     def tearDownClass(cls) -> None:
#         cls.out_dir.cleanup()
#         cls.out_dir = None
#
#     def setUp(self) -> None:
#         self.tb = None
#
#     def tearDown(self) -> None:
#         if isinstance(self.tb, DemodTopBlock):
#             self.tb.stop()
#             self.tb.wait()
