import atexit
import datetime as dt
import logging
import logging.handlers
import math
import multiprocessing as mp
import pathlib
import tempfile
import time

from collections import namedtuple as nt
from hashlib import sha256
from test import support
from typing import Union
from unittest import TestCase, skip

import ephem
import gnuradio as gr
import gnuradio.blocks
import gnuradio.analog
import gnuradio.gr
import numpy as np

from PIL import Image, ExifTags
from sats_receiver import utils
from sats_receiver.gr_modules.decoders import Decoder, AptDecoder, CcsdsConvConcatDecoder, ProtoDecoder, SatellitesDecoder, SstvDecoder
from sats_receiver.gr_modules.demodulators import FskDemod
from sats_receiver.gr_modules.epb.prober import Prober
from sats_receiver.observer import Observer
from sats_receiver.systems.apt import Apt
from sats_receiver.tle import Tle


HERE = pathlib.Path(__file__).parent
FILES = HERE / 'files'
SATYAML = HERE.parent / 'satyaml'
TIMEOUT = support.SHORT_TIMEOUT


class TestTle(Tle):
    def __init__(self):
        super().__init__({'update_period': 0})
        self.tle_file = FILES / 'test_tle.txt'
        self.fill_objects(self.tle_file, dt.datetime.now(dt.timezone.utc))

    def update_config(self, config):
        self.config = config
        return 1


class TestExecutor:
    def __init__(self):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.rd, self.wr = mp.Pipe(False)

    def execute(self, fn, *args, **kwargs):
        if self.wr:
            self.wr.send((fn, args, kwargs))

    def stop(self):
        if self.wr:
            utils.close(self.wr)
            self.wr = 0

    def action(self, t=TIMEOUT):
        try:
            x = self.rd.poll(t)
        except InterruptedError:
            x = 1
        except:
            return

        if not x:
            return

        x = self.rd.recv()
        try:
            fn, args, kwargs = x
        except ValueError:
            self.log.error('invalid task: %s', x)
            return

        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception:
                self.log.exception('%s with args=%s kwargs=%s', fn, args, kwargs)
                return


class DecoderTopBlock(gr.gr.top_block):
    def __init__(self,
                 wav_channels: int,
                 wav_fp: Union[pathlib.Path, str],
                 decoder: Decoder,
                 executor: TestExecutor,
                 freq_shift=0,
                 demod=None):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        super(DecoderTopBlock, self).__init__('DecoderTopBlock', catch_exceptions=False)

        self.executor = executor

        self.wav_src = gr.blocks.wavfile_source(str(wav_fp), False)
        self.prober = Prober()
        self.ftc = gr.blocks.float_to_complex(1)
        self.thr = gr.blocks.throttle(gr.gr.sizeof_gr_complex, decoder.samp_rate * 100)
        self.shifter = gr.blocks.rotator_cc(2.0 * math.pi * -freq_shift / decoder.samp_rate)
        self.pll = gr.analog.pll_carriertracking_cc((math.pi / 200), (2 * math.pi * (0 + 1000) / decoder.samp_rate), (2 * math.pi * (0 - 1000) / decoder.samp_rate))
        self.demod = demod
        self.decoder = decoder

        for i in range(wav_channels):
            self.connect((self.wav_src, i), (self.ftc, i))
        modules = [
            self.ftc,
            self.thr,
            # self.shifter,
            # self.pll,
        ]
        if demod:
            modules.append(self.demod)
        self.connect(*modules, self.decoder)
        self.connect(self.wav_src, self.prober)

    def start(self, max_noutput_items=10000000):
        observation_key = sha256((self.prefix + str(dt.datetime.now())).encode()).hexdigest()
        self.log.info('START')
        atexit.register(lambda x: x.stop(), self.executor)
        self.decoder.set_observation_key(observation_key)
        self.decoder.start()
        super(DecoderTopBlock, self).start(max_noutput_items)

    def stop(self):
        self.log.info('STOP')
        super(DecoderTopBlock, self).stop()

        self.decoder.finalize()
        self.executor.stop()

    def wait(self):
        super(DecoderTopBlock, self).wait()


class TestDecoders(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.sat_nt = nt('Satellite', 'name output_directory executor observer sat_ephem_tle')
        cls.rec_nt = nt('SatRecorder',
                        'satellite subname mode '
                        'sstv_sync sstv_wsr sstv_live_exec '
                        'ccc_pre_deint ccc_frame_size ccc_diff ccc_rs_dualbasis ccc_rs_interleaving ccc_derandomize '
                        'proto_deframer proto_options')

        cls.out_dir = tempfile.TemporaryDirectory('.d', 'sats-receiver-test-', ignore_cleanup_errors=True)
        cls.out_dp = pathlib.Path(cls.out_dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.out_dir.cleanup()
        cls.out_dir = None

    def setUp(self) -> None:
        self.tb = None
        self.to_close = []

        self.tle = TestTle()
        self.executor = TestExecutor()
        self.satellite = self.sat_nt(
            'TEST SAT', self.out_dp, self.executor,
            Observer(dict(latitude=11.111, longitude=-22.222, elevation=-33.333, weather=0)),
            self.tle.get('TEST SAT'),
        )
        self.recorder = self.rec_nt(self.satellite, 'test', utils.Mode.OQPSK,
                                    1, 16000, 0,
                                    1, 892, 1, 0, 4, 1,
                                    utils.ProtoDeframer.USP, {})

    def tearDown(self) -> None:
        if isinstance(self.tb, DecoderTopBlock):
            self.tb.stop()
            self.tb.wait()
        utils.close(*self.to_close)

    def test_sstv_robot36(self):
        wav_fp = FILES / 'Robot36_16kHz.wav'
        wav_samp_rate = 16000

        decoder = SstvDecoder(self.recorder, wav_samp_rate)
        self.tb = DecoderTopBlock(1, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        self.assertIsInstance(x, tuple)
        self.assertEqual(utils.Decode.SSTV, x[0])

        _, sat_name, observation_key, fn_dts = x
        self.assertEqual(self.satellite.name, sat_name)
        self.assertIsInstance(fn_dts, list)
        self.assertEqual(1, len(fn_dts))

        fn_dt = fn_dts[0]
        self.assertIsInstance(fn_dt, tuple)
        self.assertEqual(2, len(fn_dt))

        fp, d = fn_dt
        self.assertIsInstance(fp, pathlib.Path)
        self.assertTrue(fp.is_file())
        self.assertIsInstance(d, dt.datetime)

        img = Image.open(fp)
        self.to_close.append(img)
        exif = img.getexif()
        ee = exif.get_ifd(ExifTags.IFD.Exif)
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)

        self.assertEqual('Robot36', ee.get(ExifTags.Base.UserComment))
        end_time = dt.datetime.strptime(exif.get(ExifTags.Base.DateTime), '%Y:%m:%d %H:%M:%S')
        self.assertEqual(d, end_time)

        self.assertEqual(ephem.degrees(str(abs(self.satellite.observer.lat))),
                         ephem.degrees('{}:{}:{}'.format(*gps[ExifTags.GPS.GPSLatitude])))
        self.assertEqual('N', gps[ExifTags.GPS.GPSLatitudeRef])

        self.assertEqual(ephem.degrees(str(abs(self.satellite.observer.lon))),
                         ephem.degrees('{}:{}:{}'.format(*gps[ExifTags.GPS.GPSLongitude])))
        self.assertEqual('W', gps[ExifTags.GPS.GPSLongitudeRef])

        self.assertTrue(np.isclose(float(gps[ExifTags.GPS.GPSAltitude]), abs(self.satellite.observer.elev)))
        self.assertEqual(b'\1', gps[ExifTags.GPS.GPSAltitudeRef])

    def test_apt(self):
        wav_fp = FILES / 'apt_11025hz.wav'
        wav_samp_rate = 11025

        decoder = AptDecoder(self.recorder, wav_samp_rate)
        self.tb = DecoderTopBlock(1, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        self.assertIsInstance(x, tuple)
        self.assertEqual(utils.Decode.APT, x[0])

        _, res_sat_name, res_observation_key, res_filename, res_end_time = x
        self.assertEqual(self.satellite.name, res_sat_name)

        apt = Apt.from_apt(res_filename)
        self.assertEqual(self.satellite.name, apt.sat_name)
        self.assertEqual(res_end_time, apt.end_time)
        self.assertTupleEqual(self.satellite.observer.lonlat, apt.observer_lonlat)
        self.assertTupleEqual(self.tle.get_tle(self.satellite.name), apt.sat_tle)
        self.assertFalse(apt.process())

    def test_apt_fail(self):
        wav_fp = FILES / 'apt_noise_48000hz.wav'
        wav_samp_rate = 11025

        decoder = AptDecoder(self.recorder, wav_samp_rate)
        self.tb = DecoderTopBlock(1, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        with self.assertLogs(f'Apt: {self.satellite.name}', level=logging.ERROR) as e:
            x = self.tb.executor.action(TIMEOUT)
        # self.assertEqual('recording too short for telemetry decoding', e.records[0].msg)
        self.assertIsNone(x)

    def test_satellites_orbicraft_tlm(self):
        expected = bytes.fromhex('a464829c8c4060a4a6626aa6406d00f01642020001004200ea0418057607c702'
                                 '5b0b29010402000000000000000009001a004a004a0000000000ff1c25040b00'
                                 'b6a437647904110bff7a2113641eb6a4376482c10700a00f981c21de04000100'
                                 '2e00ea0418057607c7025b0b290104020000000000000000cf035a0aff1cfe1c'
                                 '00000000ff1c25040b009de99f5a7904')
        wav_fp = FILES / 'orbicraft_tlm_4800@16000.wav'
        wav_samp_rate = 16000
        cfg = dict(tlm_decode=True, file=SATYAML / 'OrbiCraft-Zorkiy.yml')

        decoder = SatellitesDecoder(self.recorder, wav_samp_rate, cfg, 1)
        self.tb = DecoderTopBlock(2, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)
        time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        self.assertIsInstance(x, tuple)
        self.assertEqual(utils.Decode.SATS, x[0])

        _, sat_name, observation_key, files = x

        self.assertIn('Telemetry', files)
        tlm_files = files['Telemetry']
        self.assertTrue(len(tlm_files) >= 2)
        for fp in tlm_files:
            self.assertIsInstance(fp, pathlib.Path)
            self.assertRegex(fp.name, r'usp_4k8.+\.(txt|bin)')
            if fp.name.endswith('.bin'):
                self.assertEqual(expected, fp.read_bytes())

    def test_satellites_geoscan_tlm_multiple(self):
        expected = [
            bytes.fromhex('848a82869e9c60a4a66460a640e103f0655e86642607f00bd4e084dd0e111609'
                          '800a07070f611de9050600000000000000000000000000000000000000000000'),
            bytes.fromhex('848a82869e9c60a4a66460a640e103f0855e86642007bf07a6e022dc0c151308'
                          '800b07070e611de9050600000000000000000000000000000000000000000000'),
            bytes.fromhex('848a82869e9c60a4a66460a640e103f0a65e86642407be0a97e054dc0b181109'
                          '800b07070e611de9050600000000000000000000000000000000000000000000'),
            bytes.fromhex('848a82869e9c60a4a66460a640e103f0e75e86642207da0488e0f0db081c0d0c'
                          '800b07070f611de9050600000000000000000000000000000000000000000000'),
        ]
        wav_fp = FILES / 'geoscan_tlm_mult_9600@48000.ogg'
        wav_samp_rate = 48000
        cfg = dict(tlm_decode=False, file=SATYAML / 'GEOSCAN-EDELVEIS.yml')

        decoder = SatellitesDecoder(self.recorder, wav_samp_rate, cfg, 0)
        self.tb = DecoderTopBlock(1, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)
        time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        self.assertIsInstance(x, tuple)
        self.assertEqual(utils.Decode.SATS, x[0])

        _, sat_name, observation_key, files = x

        self.assertIn('Telemetry', files)
        tlm_files = files['Telemetry']
        self.assertTrue(len(tlm_files) == 4)

        for i, fp in enumerate(sorted(tlm_files, key=lambda fp: fp.stat().st_mtime)):
            self.assertIsInstance(fp, pathlib.Path)
            self.assertRegex(fp.name, r'geoscan_9k6-FSK.+\.bin')
            self.assertEqual(expected[i], fp.read_bytes())

    def test_satellites_geoscan_img(self):
        wav_fp = FILES / 'geoscan_img_2023-02-02T15-18-40_9600@48000.ogg'
        wav_samp_rate = 48000
        cfg = dict(tlm_decode=True, file=SATYAML / 'GEOSCAN-EDELVEIS.yml')

        decoder = SatellitesDecoder(self.recorder, wav_samp_rate, cfg, 0)
        self.tb = DecoderTopBlock(1, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)
        time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        self.assertIsInstance(x, tuple)
        self.assertEqual(utils.Decode.SATS, x[0])

        _, sat_name, observation_key, files = x

        self.assertIn('Image', files)
        img_files = files['Image']
        self.assertTrue(len(img_files) == 1)
        for fp in img_files:
            self.assertIsInstance(fp, pathlib.Path)
            self.assertRegex(fp.name, r'GEOSCAN_.+\.jpg')

    def test_satellites_geoscan_tlm(self):
        wav_fp = FILES / 'geoscan_tlm_9600@48000.wav'
        wav_samp_rate = 48000
        cfg = dict(tlm_decode=True, file=SATYAML / 'GEOSCAN-EDELVEIS.yml')

        decoder = SatellitesDecoder(self.recorder, wav_samp_rate, cfg, 1)
        self.tb = DecoderTopBlock(2, wav_fp, decoder, self.executor)
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)
        time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        self.assertIsInstance(x, tuple)
        self.assertEqual(utils.Decode.SATS, x[0])

        _, sat_name, observation_key, files = x

        self.assertIn('Telemetry', files)
        tlm_files = files['Telemetry']
        self.assertTrue(len(tlm_files) == 2)
        for fp in tlm_files:
            self.assertIsInstance(fp, pathlib.Path)
            self.assertRegex(fp.name, r'geoscan_9k6-FSK.+\.(txt|bin)')

    def test_proto(self):
        wav_fp = FILES / 'orbicraft_tlm_4800@16000.wav'
        wav_samp_rate = 16000

        chs = [4800,]
        decoder = ProtoDecoder(self.recorder, chs)
        self.tb = DecoderTopBlock(2, wav_fp, decoder, self.executor, demod=FskDemod(wav_samp_rate, chs))
        self.tb.start()

        while self.tb.prober.changes():
            time.sleep(self.tb.prober.measure_s)
        time.sleep(self.tb.prober.measure_s)

        self.tb.stop()
        self.tb.wait()

        x = self.tb.executor.action(TIMEOUT)
        print(x)
        self.assertIsInstance(x, tuple)
        dtype, deftype, sat_name, observation_key, res_filename, end_time = x
        self.assertEqual(utils.Decode.PROTO, dtype)
        self.assertEqual(utils.ProtoDeframer.USP, deftype)
        self.assertRegex(res_filename.name, r'.+\.(kss)')
