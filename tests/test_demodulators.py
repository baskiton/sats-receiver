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

from sats_receiver.gr_modules.demodulators import GmskDemod
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
                 channels: list[int]):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        super(DemodTopBlock, self).__init__('DemodTopBlock', catch_exceptions=False)

        self.samp_rate = samp_rate
        self.channels = channels

        self.fsrc = gr.blocks.file_source(gr.gr.sizeof_gr_complex, str(fp), False, 0, samp_rate)
        self.thr = gr.blocks.throttle(gr.gr.sizeof_gr_complex, samp_rate, True)
        self.ctr = gr.blocks.complex_to_real()
        self.probe = Prober()
        self.gmsk_demod = GmskDemod(samp_rate, channels)
        self.fsinks = {}
        for rate in channels:
            fn = str(out_dir / str(rate))
            fsink = gr.blocks.file_sink(gr.gr.sizeof_float, fn, False)
            fsink.set_unbuffered(False)
            self.fsinks[rate] = fn, fsink

        self.connect(
            self.fsrc,
            self.thr,
            self.gmsk_demod,
        )
        self.connect(self.thr, self.ctr, self.probe)
        for i, rate in enumerate(channels):
            self.connect((self.gmsk_demod, i), self.fsinks[rate][1])

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


class TestDemod(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.out_dir = tempfile.TemporaryDirectory('.d', 'sats-receiver-test', ignore_cleanup_errors=True)
        cls.out_dp = pathlib.Path(cls.out_dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.out_dir.cleanup()
        cls.out_dir = None

    def setUp(self) -> None:
        self.tb = None

    def tearDown(self) -> None:
        if isinstance(self.tb, DemodTopBlock):
            self.tb.stop()
            self.tb.wait()

    def test_gmsk_9600_9600(self):
        samp_rate = 9600
        channels = [9600, 4800, 2400]
        f = FILES / 'gmsk_9600@9600_10010111.bin'
        subseq = np.array(list('10010111'), dtype=np.uint8)

        self.tb = DemodTopBlock(f, self.out_dp, samp_rate, channels)
        self.tb.start()

        while self.tb.probe.changes():
            time.sleep(self.tb.probe.measure_s)
        time.sleep(self.tb.probe.measure_s)

        self.tb.stop()
        self.tb.wait()

        f_9600 = np.fromfile(self.tb.fsinks[9600][0], dtype=np.float32).astype(np.uint8)
        f_4800 = np.fromfile(self.tb.fsinks[4800][0], dtype=np.float32).astype(np.uint8)
        f_2400 = np.fromfile(self.tb.fsinks[2400][0], dtype=np.float32).astype(np.uint8)

        rw_9600 = rolling_window(f_9600, subseq.size)
        rw_4800 = rolling_window(f_4800, subseq.size)
        rw_2400 = rolling_window(f_2400, subseq.size)

        x = np.all(rw_9600 == subseq, 1)
        self.assertTrue(np.any(x))
        y = np.argmax(x)
        x = np.resize(f_9600[y:], ((f_9600.size - y) // subseq.size, subseq.size))
        self.assertTrue(np.all(x == subseq))

        self.assertFalse(np.any(np.all(rw_4800 == subseq, 1)))
        self.assertFalse(np.any(np.all(rw_2400 == subseq, 1)))

    def test_gmsk_4800_9600(self):
        samp_rate = 9600
        channels = [9600, 4800, 2400]
        f = FILES / 'gmsk_4800@9600_10010111.bin'
        subseq = np.array(list('10010111'), dtype=np.uint8)

        self.tb = DemodTopBlock(f, self.out_dp, samp_rate, channels)
        self.tb.start()

        while self.tb.probe.changes():
            time.sleep(self.tb.probe.measure_s)
        time.sleep(self.tb.probe.measure_s)

        self.tb.stop()
        self.tb.wait()

        f_9600 = np.fromfile(self.tb.fsinks[9600][0], dtype=np.float32).astype(np.uint8)
        f_4800 = np.fromfile(self.tb.fsinks[4800][0], dtype=np.float32).astype(np.uint8)
        f_2400 = np.fromfile(self.tb.fsinks[2400][0], dtype=np.float32).astype(np.uint8)

        rw_9600 = rolling_window(f_9600[:subseq.size * 2], subseq.size)
        rw_4800 = rolling_window(f_4800[:subseq.size * 2], subseq.size)
        rw_2400 = rolling_window(f_2400[:subseq.size * 2], subseq.size)

        x = np.all(rw_4800 == subseq, 1)
        self.assertTrue(np.any(x))
        y = np.argmax(x)
        x = np.resize(f_4800[y:], ((f_4800.size - y) // subseq.size, subseq.size))
        self.assertTrue(np.all(x == subseq))

        self.assertFalse(np.any(np.all(rw_9600 == subseq, 1)))
        self.assertFalse(np.any(np.all(rw_2400 == subseq, 1)))

    def test_gmsk_2400_9600(self):
        samp_rate = 9600
        channels = [9600, 4800, 2400]
        f = FILES / 'gmsk_2400@9600_10010111.bin'
        subseq = np.array(list('10010111'), dtype=np.uint8)

        self.tb = DemodTopBlock(f, self.out_dp, samp_rate, channels)
        self.tb.start()

        while self.tb.probe.changes():
            time.sleep(self.tb.probe.measure_s)
        time.sleep(self.tb.probe.measure_s)

        self.tb.stop()
        self.tb.wait()

        f_9600 = np.fromfile(self.tb.fsinks[9600][0], dtype=np.float32).astype(np.uint8)
        f_4800 = np.fromfile(self.tb.fsinks[4800][0], dtype=np.float32).astype(np.uint8)
        f_2400 = np.fromfile(self.tb.fsinks[2400][0], dtype=np.float32).astype(np.uint8)

        rw_9600 = rolling_window(f_9600[:subseq.size * 2], subseq.size)
        rw_4800 = rolling_window(f_4800[:subseq.size * 2], subseq.size)
        rw_2400 = rolling_window(f_2400[:subseq.size * 2], subseq.size)

        x = np.all(rw_2400 == subseq, 1)
        self.assertTrue(np.any(x))
        y = np.argmax(x)
        x = np.resize(f_2400[y:], ((f_2400.size - y) // subseq.size, subseq.size))
        self.assertTrue(np.all(x == subseq))

        self.assertFalse(np.any(np.all(rw_9600 == subseq, 1)))
        self.assertFalse(np.any(np.all(rw_4800 == subseq, 1)))
