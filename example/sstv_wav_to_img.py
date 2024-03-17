import atexit
import datetime as dt
import logging
import logging.handlers
import multiprocessing as mp
import pathlib
import time

from hashlib import sha256

import gnuradio as gr
import gnuradio.blocks
import gnuradio.gr
import numpy as np
import scipy.io.wavfile as sp_wav

from sats_receiver import utils
from sats_receiver.executor import Executor
from sats_receiver.gr_modules.decoders import SstvDecoder
from sats_receiver.gr_modules.epb.prober import Prober


FILES = pathlib.Path(__file__).parent.parent / 'tests/files'


def setup_logging(q: mp.Queue, log_lvl: int):
    if not isinstance(log_lvl, int):
        raise ValueError('Invalid log level: %s' % log_lvl)

    logger = logging.getLogger()
    logger.setLevel(log_lvl)
    logger.addHandler(logging.handlers.QueueHandler(q))
    mp.get_logger().setLevel(log_lvl)

    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(name)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    # fh = logging.handlers.TimedRotatingFileHandler(LOGSDIR / 'sats_receiver.log', 'midnight')
    # fh.setFormatter(fmt)

    qhl = logging.handlers.QueueListener(q, sh)
    qhl.start()
    atexit.register(qhl.stop)

    # PIL logging level
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.DEBUG + 2)

    gr_logger = gr.gr.logging()
    gr_logger.set_default_level(gr.gr.log_levels.warn)


class DecoderTopBlock(gr.gr.top_block):
    def __init__(self,
                 wav_fp: pathlib.Path,
                 out_dp: pathlib.Path,  # out dir
                 q: mp.Queue = None):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        super(DecoderTopBlock, self).__init__('DecoderTopBlock', catch_exceptions=False)

        self.executor = Executor(q)

        samp_rate, wav_arr = sp_wav.read(wav_fp)
        self.log.debug('%s (%sHz)', wav_fp.name, utils.num_disp(samp_rate))
        if len(wav_arr.shape) > 1:
            self.log.debug('stereo to mono (first channel used)')
            wav_arr = wav_arr[0]

        wav_type = wav_arr.dtype
        if wav_arr.dtype != np.float32:
            wav_arr = wav_arr.astype(np.float32)
            x = 256
            if wav_type == np.int32:
                x = 1 << 32
            elif wav_type == np.int16:
                x = 1 << 16
            wav_arr /= (x - 1)
        wav_arr.resize(int((wav_arr.size + wav_arr.size % samp_rate) + samp_rate), refcheck=False)

        self.vector_source = gr.blocks.vector_source_f(wav_arr, False, 1, [])
        self.prober = Prober(1)
        self.ftc = gr.blocks.float_to_complex(1)
        self.decoder = SstvDecoder(
            sat_name='WavFile',
            subname='',
            samp_rate=samp_rate,
            out_dir=out_dp,
        )

        self.connect(
            self.vector_source,
            self.ftc,
            self.decoder,
        )
        self.connect(
            self.vector_source,
            self.prober,
        )

    def start(self, max_noutput_items=10000000):
        self.log.info('START')
        self.executor.start()
        atexit.register(lambda x: (x.stop(), x.join()), self.executor)
        self.decoder.start()
        super(DecoderTopBlock, self).start(max_noutput_items)

    def stop(self):
        self.log.info('STOP')
        super(DecoderTopBlock, self).stop()

        fin_key = sha256((self.prefix + str(dt.datetime.now())).encode()).hexdigest()
        self.decoder.finalize(self.executor, fin_key)

        self.executor.stop()

    def wait(self):
        super(DecoderTopBlock, self).wait()
        self.executor.join()


if __name__ == '__main__':
    q = mp.Queue()
    setup_logging(q, log_lvl=logging.DEBUG)

    out_dir = pathlib.Path('~/sats_receiver/from_wav').expanduser()
    tb = DecoderTopBlock(FILES / 'Robot36_16kHz.wav', out_dir, q)
    tb.start()

    while tb.prober.changes():
        time.sleep(tb.prober.measure_s)

    tb.stop()
    tb.wait()
