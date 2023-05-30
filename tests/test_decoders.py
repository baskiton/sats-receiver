import atexit
import datetime as dt
import logging
import logging.handlers
import multiprocessing as mp
import multiprocessing.connection
import pathlib
import tempfile
import time

from hashlib import sha256
from test import support
from unittest import TestCase

import ephem
import gnuradio as gr
import gnuradio.blocks
import gnuradio.gr
import numpy as np

from PIL import Image, ExifTags
from sats_receiver import utils
from sats_receiver.manager import Executor
from sats_receiver.gr_modules.decoders import Decoder, SstvDecoder
from sats_receiver.observer import Observer


HERE = pathlib.Path(__file__).parent
FILES = HERE / 'files'
TIMEOUT = support.SHORT_TIMEOUT


class DecoderExecutor(Executor):
    def __init__(self, ret_wr: mp.connection.Connection):
        super().__init__()
        self.ret_wr = ret_wr

    def action(self):
        while 1:
            self.sysu.collect()

            try:
                x = self.rd.poll(1)
            except InterruptedError:
                x = 1
            except:
                continue

            if not x:
                continue

            x = self.rd.recv()

            if x == '.':
                break

            try:
                fn, args, kwargs = x
            except ValueError:
                self.log.error('invalid task: %s', x)
                continue

            if callable(fn):
                try:
                    x = fn(*args, **kwargs)
                except Exception:
                    self.log.exception('%s with args=%s kwargs=%s', fn, args, kwargs)
                    continue

                if x and isinstance(x, tuple):
                    if len(x) == 4:
                        sat_name, fin_key, res_filename, end_time = x
                    elif len(x) == 3:
                        sat_name, fin_key, fn_dt = x

                self.ret_wr.send(x)


class DecoderTopBlock(gr.gr.top_block):
    def __init__(self,
                 wav_fp: pathlib.Path,
                 ret_wr: mp.connection.Connection,
                 decoder: Decoder):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        super(DecoderTopBlock, self).__init__('DecoderTopBlock', catch_exceptions=False)

        self.executor = DecoderExecutor(ret_wr)
        self.executor.start()
        atexit.register(lambda x: (x.stop(), x.join()), self.executor)

        self.blocks_wavfile_source = gr.blocks.wavfile_source(str(wav_fp), False)
        self.blocks_float_to_complex = gr.blocks.float_to_complex(1)
        self.decoder = decoder

        ##################################################
        # Connections
        ##################################################
        self.connect(
            self.blocks_wavfile_source,
            self.blocks_float_to_complex,
            self.decoder,
        )

    def start(self, max_noutput_items=10000000):
        self.log.info('START')
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


class TestDecoders(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.out_dir = tempfile.TemporaryDirectory('.d', 'sats-receiver-test', ignore_cleanup_errors=True)
        cls.out_dp = pathlib.Path(cls.out_dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.out_dir.cleanup()
        cls.out_dir = None

    def setUp(self) -> None:
        self.ret_rd, self.ret_wr = mp.Pipe(False)
        self.tb = None
        self.to_close = [self.ret_rd, self.ret_wr]

    def tearDown(self) -> None:
        if isinstance(self.tb, DecoderTopBlock):
            self.tb.stop()
            self.tb.wait()
        utils.close(*self.to_close)

    def test_sstv_robot36(self):
        lat = 11.111
        lon = -22.222
        alt = -33.333
        sstv_wsr = 16000
        wav_fp = FILES / 'Robot36_16kHz.wav'
        wav_samp_rate = 16000

        decoder = SstvDecoder(
            sat_name='Test Sat',
            samp_rate=wav_samp_rate,
            out_dir=self.out_dp,
            observer=Observer({'latitude': lat,
                               'longitude': lon,
                               'elevation': alt,
                               'weather': False}),
            do_sync=1,
            wsr=sstv_wsr,
        )
        self.tb = DecoderTopBlock(wav_fp, self.ret_wr, decoder)
        self.tb.start()
        time.sleep(TIMEOUT)
        self.tb.stop()
        self.tb.wait()

        x = self.ret_rd.poll(TIMEOUT)
        self.assertTrue(x)

        x = self.ret_rd.recv()
        self.assertIsInstance(x, tuple)
        self.assertEqual(len(x), 3)

        sat_name, fin_key, fn_dts = x
        self.assertEqual(sat_name, 'Test Sat')
        self.assertIsInstance(fn_dts, list)
        self.assertEqual(len(fn_dts), 1)

        fn_dt = fn_dts[0]
        self.assertIsInstance(fn_dt, tuple)
        self.assertEqual(len(fn_dt), 2)

        fp, d = fn_dt
        self.assertIsInstance(fp, pathlib.Path)
        self.assertTrue(fp.is_file())
        self.assertIsInstance(d, dt.datetime)

        img = Image.open(fp)
        self.to_close.append(img)
        exif = img.getexif()
        ee = exif.get_ifd(ExifTags.IFD.Exif)
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)

        self.assertEqual(ee.get(ExifTags.Base.UserComment), 'Robot36')
        end_time = dt.datetime.strptime(exif.get(ExifTags.Base.DateTime), '%Y:%m:%d %H:%M:%S')
        self.assertEqual(end_time, d)

        self.assertEqual(ephem.degrees('{}:{}:{}'.format(*gps[ExifTags.GPS.GPSLatitude])),
                         ephem.degrees(str(abs(lat))))
        self.assertEqual(gps[ExifTags.GPS.GPSLatitudeRef], 'N')

        self.assertEqual(ephem.degrees('{}:{}:{}'.format(*gps[ExifTags.GPS.GPSLongitude])),
                         ephem.degrees(str(abs(lon))))
        self.assertEqual(gps[ExifTags.GPS.GPSLongitudeRef], 'W')

        self.assertTrue(np.isclose(float(gps[ExifTags.GPS.GPSAltitude]), abs(alt)))
        self.assertEqual(gps[ExifTags.GPS.GPSAltitudeRef], b'\1')
