import datetime as dt
import logging
import math
import pathlib

from typing import Optional, Union

import dateutil.tz
import ephem
import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.fft
import gnuradio.filter
import gnuradio.gr

from PIL import ExifTags

from sats_receiver import utils
from sats_receiver.gr_modules.epb import sstv as sstv_epb
from sats_receiver.observer import Observer
from sats_receiver.systems import apt, sstv


class Decoder(gr.gr.hier_block2):
    def __init__(self,
                 name: str,
                 sat_name: str,
                 samp_rate: Union[int, float],
                 out_dir: pathlib.Path):
        self.prefix = f'{self.__class__.__name__}: {sat_name}'
        self.log = logging.getLogger(self.prefix)

        super(Decoder, self).__init__(
            name,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.now = dt.datetime.fromtimestamp(0)
        self.sat_name = sat_name
        self.samp_rate = samp_rate
        self.out_dir = out_dir
        self.tmp_file = utils.mktmp(prefix='_'.join(name.lower().split()))
        self.base_kw = dict(log=self.log, sat_name=sat_name, out_dir=out_dir)

    def start(self):
        self.tmp_file = self.out_dir / ('_'.join([self.t.strftime('%Y%m%d%H%M%S'),
                                                  *self.name().lower().split()]) + '.tmp')
        self.base_kw.update(tmp_file=self.tmp_file)

    def finalize(self, executor, fin_key: str):
        pass

    @property
    def t(self) -> dt.datetime:
        t = dt.datetime.now()
        self.now = t
        return t


class RawDecoder(Decoder):
    def __init__(self,
                 sat_name: str,
                 samp_rate: Union[int, float],
                 out_dir: pathlib.Path):
        super(RawDecoder, self).__init__('Raw Decoder', sat_name, samp_rate, out_dir)

        self.ctf = gr.blocks.complex_to_float(1)
        self.wav_sink = gr.blocks.wavfile_sink(
            str(self.tmp_file),
            2,
            samp_rate,
            gr.blocks.FORMAT_WAV,
            gr.blocks.FORMAT_FLOAT,
            False
        )
        self.wav_sink.close()
        utils.unlink(self.tmp_file)

        self.connect(self, self.ctf, self.wav_sink)
        self.connect((self.ctf, 1), (self.wav_sink, 1))

    def start(self):
        super(RawDecoder, self).start()

        self.wav_sink.open(str(self.tmp_file))

    def finalize(self, executor, fin_key: str):
        self.wav_sink.close()
        if self.tmp_file.exists():
            executor.execute(self._raw_finalize, **self.base_kw, fin_key=fin_key)

    @staticmethod
    def _raw_finalize(log, sat_name, out_dir, tmp_file, fin_key) -> Optional[tuple[str, str, str, dt.datetime]]:
        log.debug('finalizing...')

        d = dt.datetime.fromtimestamp(tmp_file.stat().st_mtime, dateutil.tz.tzutc())
        res_fn = tmp_file.rename(out_dir / d.strftime(f'{sat_name}_%Y-%m-%d_%H-%M-%S_RAW.wav'))
        st = res_fn.stat()
        log.info('finish: %s (%s)', res_fn, utils.numbi_disp(st.st_size))

        return sat_name, fin_key, res_fn, dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc())


class AptDecoder(Decoder):
    def __init__(self,
                 sat_name: str,
                 samp_rate: Union[int, float],
                 out_dir: pathlib.Path,
                 sat_ephem_tle: Optional[tuple[ephem.EarthSatellite, tuple[str, str, str]]],
                 observer_lonlat: tuple[float, float]):
        name = 'APT Decoder'
        super(AptDecoder, self).__init__(name, sat_name, samp_rate, out_dir)

        pfx = '_'.join(name.lower().split())
        self.corr_file = utils.mktmp(dir=out_dir, prefix=pfx, suffix='.corr')
        self.peaks_file = utils.mktmp(dir=out_dir, prefix=pfx, suffix='.peaks')
        self.sat_ephem_tle = sat_ephem_tle
        self.observer_lonlat = observer_lonlat
        self.base_kw.update(sat_tle=sat_ephem_tle[1], observer_lonlat=observer_lonlat)

        resamp_gcd = math.gcd(samp_rate, apt.Apt.WORK_RATE)

        self.frs = gr.blocks.rotator_cc(2 * math.pi * -apt.Apt.CARRIER_FREQ / samp_rate)
        self.rsp = gr.filter.rational_resampler_ccc(
            interpolation=apt.Apt.WORK_RATE // resamp_gcd,
            decimation=samp_rate // resamp_gcd,
            taps=[],
            fractional_bw=0,
        )

        tofilter_freq = 2080
        lp_gcd = math.gcd(apt.Apt.WORK_RATE, tofilter_freq)

        self.lpf = gr.filter.fir_filter_ccf(
            1,
            gr.filter.firdes.low_pass(
                gain=1,
                sampling_freq=apt.Apt.WORK_RATE / lp_gcd,
                cutoff_freq=tofilter_freq / lp_gcd,
                transition_width=0.1,
                window=gr.fft.window.WIN_HAMMING,
                param=6.76,
            )
        )
        self.ctm = gr.blocks.complex_to_mag()
        self.ftc = gr.blocks.float_to_complex()
        # self.dc_rem = gr.blocks.correctiq()
        self.correllator = gr.digital.corr_est_cc(
            symbols=apt.Apt.SYNC_A,
            sps=1,
            mark_delay=1,
            threshold=0.9,
            threshold_method=gr.digital.THRESHOLD_ABSOLUTE,
        )
        self.ctf_out = gr.blocks.complex_to_float()
        self.ctf_corr = gr.blocks.complex_to_float()

        self.out_file_sink = gr.blocks.file_sink(gr.gr.sizeof_float, str(self.tmp_file), False)
        self.out_corr_sink = gr.blocks.file_sink(gr.gr.sizeof_float, str(self.corr_file), False)
        self.peak_detector = gr.blocks.peak_detector2_fb(
            threshold_factor_rise=7,
            look_ahead=apt.Apt.FRAME_WIDTH * apt.Apt.WORK_RATE // apt.Apt.FINAL_RATE,
            alpha=0.001,
        )
        self.out_peaks_sink = gr.blocks.file_sink(gr.gr.sizeof_char, str(self.peaks_file), False)

        self.connect(
            self,
            self.frs,
            self.rsp,
            self.lpf,
            self.ctm,
            self.ftc,
            # self.dc_rem,
            self.correllator,
            self.ctf_out,
            self.out_file_sink,
        )
        self.connect(
            (self.correllator, 1),
            self.ctf_corr,
            self.out_corr_sink,
        )
        self.connect(
            self.ctf_corr,
            self.peak_detector,
            self.out_peaks_sink,
        )

        utils.close(self.out_file_sink, self.out_corr_sink, self.out_peaks_sink)
        utils.unlink(self.tmp_file, self.corr_file, self.peaks_file)

    def start(self):
        super(AptDecoder, self).start()
        self.corr_file = self.out_dir / ('_'.join([self.now.strftime('%Y%m%d%H%M%S'),
                                                   *self.name().lower().split()]) + '.corr.tmp')
        self.peaks_file = self.out_dir / ('_'.join([self.now.strftime('%Y%m%d%H%M%S'),
                                                    *self.name().lower().split()]) + '.peaks.tmp')
        self.base_kw.update(corr_file=self.corr_file, peaks_file=self.peaks_file)

        self.out_file_sink.open(str(self.tmp_file))
        self.out_file_sink.set_unbuffered(False)

        self.out_corr_sink.open(str(self.corr_file))
        self.out_corr_sink.set_unbuffered(False)

        self.out_peaks_sink.open(str(self.peaks_file))
        self.out_peaks_sink.set_unbuffered(False)

    def finalize(self, executor, fin_key: str):
        self.out_file_sink.do_update()
        self.out_corr_sink.do_update()
        self.out_peaks_sink.do_update()

        utils.close(self.out_file_sink, self.out_corr_sink, self.out_peaks_sink)

        for p in self.tmp_file, self.corr_file, self.peaks_file:
            if not p.exists():
                self.log.warning('%s: missing components `%s`', p)
                utils.unlink(self.tmp_file, self.corr_file, self.peaks_file)
                return

        executor.execute(self._apt_finalize, **self.base_kw, fin_key=fin_key)

    @staticmethod
    def _apt_finalize(log: logging.Logger,
                      sat_name: str,
                      sat_tle: tuple[str, str, str],
                      observer_lonlat: tuple[float, float],
                      tmp_file: pathlib.Path,
                      corr_file: pathlib.Path,
                      peaks_file: pathlib.Path,
                      out_dir: pathlib.Path,
                      fin_key: str) -> Optional[tuple[str, str, pathlib.Path, dt.datetime]]:
        log.debug('finalizing...')

        a = apt.Apt(sat_name, tmp_file, corr_file, peaks_file, sat_tle, observer_lonlat)
        x = a.process()

        utils.unlink(tmp_file, corr_file, peaks_file)

        if x:
            log.info('finish with error')
        else:
            res_fn, sz = a.to_apt(out_dir)
            log.info('finish: %s (%s)', res_fn, utils.numbi_disp(sz))

            return sat_name, fin_key, res_fn, a.end_time


class RawStreamDecoder(Decoder):
    def __init__(self,
                 sat_name: str,
                 samp_rate: Union[int, float],
                 out_dir: pathlib.Path,
                 name='RAW Stream Decoder'):
        super(RawStreamDecoder, self).__init__(name, sat_name, samp_rate, out_dir)

        self.ctf = gr.blocks.complex_to_float(1)
        self.rail = gr.analog.rail_ff(-1, 1)
        self.ftch = gr.blocks.float_to_char(1, 127)
        # self.fts = gr.blocks.float_to_short(1, 32767)
        # self.stch = gr.blocks.short_to_char(1)

        self.out_file_sink = gr.blocks.file_sink(gr.gr.sizeof_char, str(self.tmp_file), False)
        self.out_file_sink.close()
        utils.unlink(self.tmp_file)

        self.connect(
            self,
            self.ctf,
            self.rail,
            self.ftch,
            self.out_file_sink,
        )

    def start(self):
        super(RawStreamDecoder, self).start()

        self.out_file_sink.open(str(self.tmp_file))
        self.out_file_sink.set_unbuffered(False)

    def finalize(self, executor, fin_key: str):
        self.out_file_sink.do_update()
        self.out_file_sink.close()
        if self.tmp_file.exists():
            executor.execute(self._raw_stream_finalize, **self.base_kw, fin_key=fin_key)

    @staticmethod
    def _raw_stream_finalize(log: logging.Logger,
                             sat_name: str,
                             out_dir: pathlib.Path,
                             tmp_file: pathlib.Path,
                             fin_key: str) -> Optional[tuple[str, str, pathlib.Path, dt.datetime]]:
        log.debug('finalizing...')

        d = dt.datetime.fromtimestamp(tmp_file.stat().st_mtime, dateutil.tz.tzutc())
        res_fn = tmp_file.rename(out_dir / d.strftime(f'{sat_name}_%Y-%m-%d_%H-%M-%S_RAW.s'))
        st = res_fn.stat()
        log.info('finish: %s (%s)', res_fn, utils.numbi_disp(st.st_size))

        return sat_name, fin_key, res_fn, dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc())


class LrptDecoder(RawStreamDecoder):
    def __init__(self,
                 sat_name: str,
                 samp_rate: Union[int, float],
                 out_dir: pathlib.Path):
        raise NotImplementedError()
        super(LrptDecoder, self).__init__(sat_name, samp_rate, out_dir, name='LRPT Decoder')

    def start(self):
        super(LrptDecoder, self).start()

    def finalize(self, executor, fin_key: str):
        super(LrptDecoder, self).finalize(executor, fin_key)


class SstvDecoder(Decoder):
    _FREQ_1 = (1900 - 1200) / 2

    def __init__(self,
                 sat_name: str,
                 samp_rate: Union[int, float],
                 out_dir: pathlib.Path,
                 observer: Observer = None,
                 do_sync=True,
                 wsr=16000):
        super(SstvDecoder, self).__init__('SSTV Decoder', sat_name, samp_rate, out_dir)

        self.observer = observer
        self.do_sync = do_sync
        self.wsr = wsr

        hdr_pix_width = int(wsr * sstv.Sstv.HDR_PIX_S)
        hdr = sstv.Sstv.HDR_SYNC_WORD.repeat(hdr_pix_width)
        resamp_gcd = math.gcd(wsr, samp_rate)

        self.bpf_input = gr.filter.fir_filter_ccc(
            1,
            gr.filter.firdes.complex_band_pass(
                1,          # gain
                samp_rate,  # sampling_freq
                900,        # low_cutoff_freq
                2500,       # high_cutoff_freq
                200,        # transition_width
                gr.fft.window.WIN_HAMMING,
                6.76))
        self.frs = gr.blocks.rotator_cc(2 * math.pi * -(1900 - self._FREQ_1) / samp_rate)
        self.quad_demod = gr.analog.quadrature_demod_cf((samp_rate / (2 * math.pi * self._FREQ_1)))
        self.rsp = gr.filter.rational_resampler_fcc(
                interpolation=wsr // resamp_gcd,
                decimation=samp_rate // resamp_gcd,
                taps=[],
                fractional_bw=0
        )
        self.correllator = gr.digital.corr_est_cc(hdr, hdr_pix_width, 1, 0.4, gr.digital.THRESHOLD_ABSOLUTE)
        self.ctf_out = gr.blocks.complex_to_float()
        self.out_add_const = gr.blocks.add_const_ff(450 / self._FREQ_1)
        self.out_multiply_const = gr.blocks.multiply_const_ff(self._FREQ_1 / (750 + 450))
        self.ctf_corr = gr.blocks.complex_to_float()
        self.corr_peak_detector = gr.blocks.peak_detector2_fb(0.1, hdr.size, 0.001)
        self.sstv_epb = sstv_epb.SstvEpb(wsr, do_sync, self.log, sat_name, out_dir)

        self.connect(
            self,
            self.bpf_input,
            self.frs,
            self.quad_demod,
            self.rsp,
            self.correllator,
            self.ctf_out,
            self.out_add_const,
            self.out_multiply_const,
            (self.sstv_epb, self.sstv_epb.OUT_IN),
        )
        self.connect(
            (self.correllator, 1),
            self.ctf_corr,
            self.corr_peak_detector,
            (self.sstv_epb, self.sstv_epb.PEAKS_IN),
        )

        if observer is None:
            latlonalt = None
        else:
            latlonalt = str(observer.lat), str(observer.lon), observer.elev
        self.base_kw.update(observer_latlonalt=latlonalt)

    def start(self):
        # super(SstvDecoder, self).start()
        utils.unlink(self.tmp_file)

    def finalize(self, executor, fin_key: str):
        self.sstv_epb.stop()
        sstv_rr: list[sstv.SstvRecognizer] = self.sstv_epb.finalize()
        executor.execute(self._sstv_finalize, **self.base_kw, sstv_rr=sstv_rr, fin_key=fin_key)

    @staticmethod
    def _sstv_finalize(log: logging.Logger,
                       sat_name: str,
                       out_dir: pathlib.Path,
                       observer_latlonalt: Optional[tuple[str, str, Union[int, float]]],
                       sstv_rr: list[sstv.SstvRecognizer],
                       fin_key: str) -> tuple[str, str, list[tuple[pathlib.Path, dt.datetime]]]:
        log.debug('finalizing...')

        if observer_latlonalt:
            observer = ephem.Observer()
            observer.lat, observer.lon, observer.elev = observer_latlonalt

        fn_dt = []
        sz_sum = 0
        for i in sstv_rr:
            img = i.get_image()
            if not img:
                continue

            if observer_latlonalt:
                log.debug('add GPSInfo EXIF')
                img = utils.img_add_exif(img, observer=observer)
            exif = img.getexif()

            sstv_mode = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.UserComment, 'SSTV')
            end_time = exif.get(ExifTags.Base.DateTime)
            end_time = (dt.datetime.strptime(end_time, '%Y:%m:%d %H:%M:%S')
                        if end_time
                        else dt.datetime.utcnow())
            res_fn = out_dir / end_time.strftime(f'{sat_name}_{sstv_mode}_%Y-%m-%d_%H-%M-%S,%f.png')

            img.save(res_fn, exif=exif)

            fn_dt.append((res_fn, end_time))
            sz = res_fn.stat().st_size
            sz_sum += sz
            log.info('finish: %s (%s)', res_fn, utils.numbi_disp(sz))

        log.info('finish %s images (%s)', len(fn_dt), utils.numbi_disp(sz_sum))

        return sat_name, fin_key, fn_dt
