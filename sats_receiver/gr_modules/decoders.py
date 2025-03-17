import datetime as dt
import logging
import math
import pathlib

from typing import List, Optional, Union

import dateutil.tz
import ephem
import hashlib
import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.fft
import gnuradio.filter
import gnuradio.gr
import gnuradio.pdu
import pmt
import satellites
import satellites.components
import satellites.components.deframers
import satellites.hier
import satellites.hier.ccsds_viterbi

from PIL import ExifTags

from sats_receiver import utils
from sats_receiver.gr_modules.epb import sstv as sstv_epb
from sats_receiver.gr_modules.epb import pdu_to_cadu, deint
from sats_receiver.observer import Observer
from sats_receiver.systems import apt, satellites as sats, sstv


class Decoder(gr.gr.hier_block2):
    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float],
                 name: str,
                 dtype: utils.Decode,
                 io_signatures: List = None):
        self.prefix = f'{self.__class__.__name__}: {recorder.satellite.name}'
        self.log = logging.getLogger(self.prefix)

        if io_signatures is None:
            io_signatures = [
                gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
                gr.gr.io_signature(0, 0, 0),
            ]
        super(Decoder, self).__init__(
            name,
            *io_signatures,
        )

        self.now = dt.datetime.fromtimestamp(0)
        self.observation_key = ''
        self.recorder = recorder
        self.sat_name = recorder.satellite.name
        self.subname = recorder.subname and '_' + recorder.subname
        self.samp_rate = samp_rate
        self.out_dir = recorder.satellite.output_directory
        self.tmp_file = utils.mktmp(prefix='_'.join(name.lower().split()))
        self.base_kw = dict(log=self.log, sat_name=self.sat_name, subname=self.subname,
                            samp_rate=samp_rate, out_dir=self.out_dir, dtype=dtype,
                            observation_key='')

    def set_observation_key(self, observation_key: str):
        self.observation_key = observation_key
        self.base_kw.update(observation_key=observation_key)

    def lock_reconf(self, detach=0):
        pass

    def start(self):
        pfx = '_'.join([*self.name().lower().split(), self.t.strftime('%Y%m%d%H%M%S')])
        self.tmp_file = utils.mktmp(self.out_dir, pfx)
        self.base_kw.update(tmp_file=self.tmp_file, start_dt=self.t)

    def finalize(self):
        pass

    @property
    def t(self) -> dt.datetime:
        t = dt.datetime.now()
        self.now = t
        return t


class RawDecoder(Decoder):
    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float],
                 force_nosend_iq=False,
                 iq_in=True,
                 out_fmt=None,
                 out_subfmt=None,
                 name='Raw Decoder',
                 dtype=utils.Decode.RAW):
        super(RawDecoder, self).__init__(
            recorder,
            samp_rate,
            name,
            dtype,
            [
                gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex)
                if iq_in
                else gr.gr.io_signature(1, 1, gr.gr.sizeof_float),
                gr.gr.io_signature(0, 0, 0),
            ],
        )

        if not out_fmt:
            out_fmt = recorder.raw_out_format
        if not out_subfmt:
            out_subfmt = recorder.raw_out_subformat

        self.base_kw['wf_cfg'] = recorder.iq_waterfall
        self.base_kw['send_iq'] = recorder.iq_dump or (not force_nosend_iq and out_fmt != utils.RawOutFormat.NONE)
        if out_fmt == utils.RawOutFormat.NONE:
            out_fmt = utils.RawOutFormat.WAV
            out_subfmt = utils.RawOutDefaultSub.WAV.value

        self.base_kw['out_fmt'] = out_fmt
        self.base_kw['iq_in'] = iq_in
        self.base_kw['wf_minmax'] = recorder.satellite.receiver.wf_minmax

        self.pre_sink = self
        if iq_in:
            self.pre_sink = self.ctf = gr.blocks.complex_to_float(1)
            self.connect(self, self.pre_sink)

        ch_n = iq_in and 2 or 1
        self.wav_sink_kw = dict(
            filename=str(self.tmp_file),
            n_channels=ch_n,
            sample_rate=samp_rate,
            format=out_fmt.value,
            subformat=out_subfmt.value,
            append=False,
        )
        self.make_new_sink()

    def make_new_sink(self):
        # NOTE: only in locked state!
        self.wav_sink = gr.blocks.wavfile_sink(**self.wav_sink_kw)
        self.wav_sink.close()
        utils.unlink(self.tmp_file)
        for ch in range(self.wav_sink_kw['n_channels']):
            self.connect((self.pre_sink, ch), (self.wav_sink, ch))

    def lock_reconf(self, detach=0):
        self.wav_sink.close()
        if detach:
            return

        self.wav_sink.set_append(1)
        if not self.wav_sink.open(str(self.tmp_file)):
            # fully renew sink
            self.finalize()
            self.disconnect(self.wav_sink)
            self.start(1)

    def start(self, remake_sink=0):
        super(RawDecoder, self).start()
        if remake_sink:
            self.make_new_sink()
        self.wav_sink.open(str(self.tmp_file))

    def finalize(self):
        self.wav_sink.close()
        if self.tmp_file.exists():
            self.recorder.satellite.executor.execute(self._raw_finalize, **self.base_kw.copy())

    @staticmethod
    def _raw_finalize(log: logging.Logger,
                      sat_name: str,
                      subname: str,
                      out_dir: pathlib.Path,
                      dtype: utils.Decode,
                      tmp_file: pathlib.Path,
                      observation_key: str,
                      wf_cfg: dict,
                      send_iq: bool,
                      out_fmt: utils.RawOutFormat,
                      iq_in: bool,
                      wf_minmax: list[Union[int, float, None]],
                      **kw) -> tuple[utils.Decode, str, str, dict[utils.RawFileType, pathlib.Path], list[Union[int, float, None]], dt.datetime]:
        log.debug('finalizing...')

        st = tmp_file.stat()
        d = dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc())
        suff = 'ogg' if out_fmt == utils.RawOutFormat.OGG else 'wav'
        res_fn = tmp_file.rename(out_dir / d.strftime(f'{sat_name}_%Y-%m-%d_%H-%M-%S,%f{subname}_RAW.{suff}'))
        files = {}

        if st.st_size and wf_cfg is not None and out_fmt in (utils.RawOutFormat.WAV, utils.RawOutFormat.WAV64):
            wfp = res_fn.with_suffix('.wfc')
            try:
                wf = utils.Waterfall.from_wav(res_fn, end_timestamp=st.st_mtime, **wf_cfg)
                files[utils.RawFileType.WFC] = wf.to_cfile(wfp)
            except Exception as e:
                log.warning('WF error: %s', e)
                utils.unlink(wfp)

        if send_iq and st.st_size:
            k = utils.RawFileType.IQ if iq_in else utils.RawFileType.AUDIO
            files[k] = res_fn
        else:
            res_fn.unlink(True)

        log.info('finish: %s (%s)', files, utils.numbi_disp(st.st_size))
        if not files:
            return utils.Decode.NONE,

        return (dtype, sat_name, observation_key, files, wf_minmax,
                dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc()))


class AptDecoder(Decoder):
    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float]):
        name = 'APT Decoder'
        super(AptDecoder, self).__init__(recorder, samp_rate, name, utils.Decode.APT)

        self.already_fins = 0

        pfx = '_'.join(name.lower().split())
        self.corr_file = utils.mktmp(dir=self.out_dir, prefix=pfx, suffix='.corr')
        self.peaks_file = utils.mktmp(dir=self.out_dir, prefix=pfx, suffix='.peaks')
        self.sat_ephem_tle = recorder.satellite.sat_ephem_tle
        self.observer_lonlat = recorder.satellite.observer.lonlat
        self.base_kw.update(sat_tle=self.sat_ephem_tle[1], observer_lonlat=self.observer_lonlat)

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
        self.ctr_out = gr.blocks.complex_to_real()
        self.ctr_corr = gr.blocks.complex_to_real()

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
            self.ctr_out,
            self.out_file_sink,
        )
        self.connect(
            (self.correllator, 1),
            self.ctr_corr,
            self.out_corr_sink,
        )
        self.connect(
            self.ctr_corr,
            self.peak_detector,
            self.out_peaks_sink,
        )

        utils.close(self.out_file_sink, self.out_corr_sink, self.out_peaks_sink)
        utils.unlink(self.tmp_file, self.corr_file, self.peaks_file)

    def start(self):
        self.already_fins = 0
        super(AptDecoder, self).start()

        pfx = '_'.join([*self.name().lower().split(), self.now.strftime('%Y%m%d%H%M%S')])
        self.corr_file = utils.mktmp(self.out_dir, pfx, '.corr.tmp')
        self.peaks_file = utils.mktmp(self.out_dir, pfx, '.peaks.tmp')
        self.base_kw.update(corr_file=self.corr_file, peaks_file=self.peaks_file)

        self.out_file_sink.open(str(self.tmp_file))
        self.out_file_sink.set_unbuffered(False)

        self.out_corr_sink.open(str(self.corr_file))
        self.out_corr_sink.set_unbuffered(False)

        self.out_peaks_sink.open(str(self.peaks_file))
        self.out_peaks_sink.set_unbuffered(False)

    def finalize(self):
        if self.already_fins:
            self.log.debug('Already finalized. Skip')
            return

        self.already_fins = 1
        self.out_file_sink.do_update()
        self.out_corr_sink.do_update()
        self.out_peaks_sink.do_update()

        utils.close(self.out_file_sink, self.out_corr_sink, self.out_peaks_sink)

        for p in self.tmp_file, self.corr_file, self.peaks_file:
            if not p.exists():
                self.log.warning('missing components `%s`', p)
                utils.unlink(self.tmp_file, self.corr_file, self.peaks_file)
                return

        self.recorder.satellite.executor.execute(self._apt_finalize, **self.base_kw)

    @staticmethod
    def _apt_finalize(log: logging.Logger,
                      sat_name: str,
                      subname: str,
                      sat_tle: tuple[str, str, str],
                      observer_lonlat: tuple[float, float],
                      tmp_file: pathlib.Path,
                      corr_file: pathlib.Path,
                      peaks_file: pathlib.Path,
                      out_dir: pathlib.Path,
                      dtype: utils.Decode,
                      observation_key: str,
                      **kw) -> Optional[tuple[utils.Decode, str, str, pathlib.Path, dt.datetime]]:
        log.debug('finalizing...')

        a = apt.Apt(sat_name, tmp_file, corr_file, peaks_file, sat_tle, observer_lonlat)
        x = a.process()

        utils.unlink(tmp_file, corr_file, peaks_file)

        if x:
            log.info('finish with error')
        else:
            res_fn, sz = a.to_apt(out_dir)
            if subname:
                res_fn = res_fn.rename(res_fn.with_stem(res_fn.stem + subname))
            log.info('finish: %s (%s)', res_fn, utils.numbi_disp(sz))
            if not sz:
                res_fn.unlink(True)
                return utils.Decode.NONE,

            return dtype, sat_name, observation_key, res_fn, a.end_time


class ConstelSoftDecoder(Decoder):
    CONSTELLS = {
        # '16QAM': gr.digital.constellation_16qam,
        # '8PSK': gr.digital.constellation_8psk,
        # '8PSK_NATURAL': gr.digital.constellation_8psk_natural,
        # 'BPSK': gr.digital.constellation_bpsk,
        # 'DQPSK': gr.digital.constellation_dqpsk,
        # 'PSK': gr.digital.constellation_psk,
        utils.Mode.QPSK: gr.digital.constellation_qpsk,
        utils.Mode.OQPSK: gr.digital.constellation_qpsk,
    }

    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float],
                 name='Constellation Soft Decoder',
                 dtype=utils.Decode.CSOFT):
        super(ConstelSoftDecoder, self).__init__(recorder, samp_rate, name, dtype)
        self.base_kw['suff'] = 's'

        self.constell_mode = recorder.mode
        self.constellation = self.CONSTELLS[self.constell_mode]().base()
        self.constellation.gen_soft_dec_lut(8)
        self.constel_soft_decoder = gr.digital.constellation_soft_decoder_cf(self.constellation)
        self.rail = gr.analog.rail_ff(-1, 1)
        self.ftch = gr.blocks.float_to_char(1, 127)

        self.out_file_sink = gr.blocks.file_sink(gr.gr.sizeof_char, str(self.tmp_file), False)
        self.out_file_sink.close()
        utils.unlink(self.tmp_file)

        self.connect(
            self,
            self.constel_soft_decoder,
            self.rail,
            self.ftch,
            self.out_file_sink,
        )

    def start(self):
        super(ConstelSoftDecoder, self).start()

        if self.out_file_sink:
            self.out_file_sink.open(str(self.tmp_file))
            self.out_file_sink.set_unbuffered(False)

    def finalize(self):
        if self.out_file_sink:
            self.out_file_sink.do_update()
            self.out_file_sink.close()
        if self.tmp_file.exists():
            self.recorder.satellite.executor.execute(self._constel_soft_finalize, **self.base_kw)

    @staticmethod
    def _constel_soft_finalize(log: logging.Logger,
                               sat_name: str,
                               subname: str,
                               out_dir: pathlib.Path,
                               dtype: utils.Decode,
                               tmp_file: pathlib.Path,
                               observation_key: str,
                               suff: str,
                               **kw) -> tuple[utils.Decode, str, str, pathlib.Path, dt.datetime]:
        log.debug('finalizing...')

        st = tmp_file.stat()
        d = dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc())
        res_fn = tmp_file.rename(out_dir / d.strftime(f'{sat_name}_%Y-%m-%d_%H-%M-%S,%f{subname}.{suff}'))
        log.info('finish: %s (%s)', res_fn, utils.numbi_disp(st.st_size))
        if not st.st_size:
            res_fn.unlink(True)
            return utils.Decode.NONE,

        return dtype, sat_name, observation_key, res_fn, dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc())


class CcsdsConvConcatDecoder(ConstelSoftDecoder):
    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float]):
        super().__init__(recorder, samp_rate, 'CCSDS Conv Concat Decoder', utils.Decode.CCSDSCC)
        self.base_kw['suff'] = 'cadu'

        is_qpsk = self.constell_mode.name.endswith('QPSK')
        frame_size = recorder.ccc_frame_size or 892
        pre_deint = recorder.ccc_pre_deint
        diff = recorder.ccc_diff and 'differential' or None
        rs_dualbasis = recorder.ccc_rs_dualbasis and 'dual' or 'conventional'
        rs_interleaving = recorder.ccc_rs_interleaving or 4
        derandomize = recorder.ccc_derandomize and 'CCSDS' or 'none'

        self.disconnect(
            self.rail,
            self.ftch,
            self.out_file_sink,
        )
        del self.ftch
        self.out_file_sink = 0

        self.mul = gr.blocks.multiply_const_ff(not pre_deint or 127, 1)
        self.deframer0 = satellites.components.deframers.ccsds_concatenated_deframer(
            frame_size=frame_size,
            precoding=diff,
            rs_basis=rs_dualbasis,
            rs_interleaving=rs_interleaving,
            scrambler=derandomize,
            convolutional='CCSDS',
            syncword_threshold=4,
        )
        if is_qpsk:
            self.deframer1 = satellites.components.deframers.ccsds_concatenated_deframer(
                frame_size=892,
                precoding=diff,
                rs_basis=rs_dualbasis,
                rs_interleaving=rs_interleaving,
                scrambler=derandomize,
                convolutional='CCSDS uninverted',
                syncword_threshold=4,
            )
        self.out = pdu_to_cadu.PduToCadu(b'\x1A\xCF\xFC\x1D', 1024)

        self.connect(
            self.rail,
            self.mul,
        )

        if pre_deint:
            # self.stopdu1 = satellites.hier.sync_to_pdu_soft(
            #     packlen=72,
            #     sync='00100111',
            #     threshold=1,
            # )
            # self.stopdu2 = satellites.hier.sync_to_pdu_soft(
            #     packlen=72,
            #     sync='01001110',
            #     threshold=1,
            # )
            self.deint = deint.Deinterleave()
            self.pdutos = gr.pdu.pdu_to_stream_f(gr.pdu.EARLY_BURST_APPEND, 64)

            self.connect(self.mul, self.deint)
            self.msg_connect((self.deint, 'out'), (self.pdutos, 'pdus'))
            self.connect(self.pdutos, self.deframer0)
            if is_qpsk:
                self.connect(self.pdutos, self.deframer1)
        else:
            self.connect(self.mul, self.deframer0)
            if is_qpsk:
                self.connect(self.mul, self.deframer1)

        self.msg_connect((self.deframer0, 'out'), (self.out, 'in'))
        if is_qpsk:
            self.msg_connect((self.deframer1, 'out'), (self.out, 'in'))

    def start(self):
        super().start()
        self.out.set_out_f(self.tmp_file)


class SstvDecoder(Decoder):
    _FREQ_1 = (1900 - 1200) / 2

    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float]):
        super(SstvDecoder, self).__init__(recorder, samp_rate, 'SSTV Decoder', utils.Decode.SSTV)

        self.observer = recorder.satellite.observer
        self.do_sync = recorder.sstv_sync
        self.wsr = recorder.sstv_wsr or 16000
        self.live_exec = recorder.sstv_live_exec

        hdr_pix_width = int(self.wsr * sstv.Sstv.HDR_PIX_S)
        hdr = sstv.Sstv.HDR_SYNC_WORD.repeat(hdr_pix_width)
        resamp_gcd = math.gcd(self.wsr, samp_rate)

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
                interpolation=self.wsr // resamp_gcd,
                decimation=samp_rate // resamp_gcd,
                taps=[],
                fractional_bw=0
        )
        self.correllator = gr.digital.corr_est_cc(hdr, hdr_pix_width, 1, 0.4, gr.digital.THRESHOLD_ABSOLUTE)
        self.ctr_out = gr.blocks.complex_to_real()
        self.out_add_const = gr.blocks.add_const_ff(450 / self._FREQ_1)
        self.out_multiply_const = gr.blocks.multiply_const_ff(self._FREQ_1 / (750 + 450))
        self.ctr_corr = gr.blocks.complex_to_real()
        self.corr_peak_detector = gr.blocks.peak_detector2_fb(0.1, hdr.size, 0.001)
        if self.live_exec:
            self.live_exec = self.recorder.satellite.executor, (self._sstv_finalize,), self.base_kw
        self.sstv_epb = sstv_epb.SstvEpb(self.wsr, self.do_sync, self.log, self.sat_name, self.out_dir, self.live_exec)

        self.connect(
            self,
            self.bpf_input,
            self.frs,
            self.quad_demod,
            self.rsp,
            self.correllator,
            self.ctr_out,
            self.out_add_const,
            self.out_multiply_const,
            (self.sstv_epb, self.sstv_epb.OUT_IN),
        )
        self.connect(
            (self.correllator, 1),
            self.ctr_corr,
            self.corr_peak_detector,
            (self.sstv_epb, self.sstv_epb.PEAKS_IN),
        )

        if self.observer is None:
            latlonalt = None
        else:
            latlonalt = str(self.observer.lat), str(self.observer.lon), self.observer.elev
        self.base_kw.update(observer_latlonalt=latlonalt)

    def start(self):
        # super(SstvDecoder, self).start()
        utils.unlink(self.tmp_file)

    def finalize(self):
        self.sstv_epb.stop()
        sstv_rr: list[sstv.SstvRecognizer] = self.sstv_epb.finalize()
        self.recorder.satellite.executor.execute(self._sstv_finalize, **self.base_kw, sstv_rr=sstv_rr)

    @staticmethod
    def _sstv_finalize(log: logging.Logger,
                       sat_name: str,
                       subname: str,
                       out_dir: pathlib.Path,
                       dtype: utils.Decode,
                       observer_latlonalt: Optional[tuple[str, str, Union[int, float]]],
                       sstv_rr: list[sstv.SstvRecognizer],
                       observation_key: str,
                       **kw) -> tuple[utils.Decode, str, str, list[tuple[pathlib.Path, dt.datetime]]]:
        log.debug('finalizing...')

        if observer_latlonalt:
            observer = ephem.Observer()
            observer.lat, observer.lon, observer.elev = observer_latlonalt

        fn_dt = []
        sz_sum = 0
        for i in sstv_rr:
            if not i.sstv and i.vis_code:
                continue
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
            img_hash = hashlib.sha256(img.tobytes()).hexdigest()[:8]
            res_fn = out_dir / end_time.strftime(f'{sat_name}_{sstv_mode}_%Y-%m-%d_%H-%M-%S{subname}_{img_hash}.png')

            img.save(res_fn, exif=exif)

            fn_dt.append((res_fn, end_time))
            sz = res_fn.stat().st_size
            sz_sum += sz
            log.info('finish: %s (%s)', res_fn, utils.numbi_disp(sz))

        log.info('finish %s images (%s)', len(fn_dt), utils.numbi_disp(sz_sum))

        return dtype, sat_name, observation_key, fn_dt


class SatellitesDecoder(Decoder):
    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float],
                 config: dict,
                 is_iq=True):
        super(SatellitesDecoder, self).__init__(recorder, samp_rate, 'GR Satellites Decoder', utils.Decode.SATS)
        opt_str = (
            f' --file_output_path="{self.out_dir}"'
            # f' --codec2_ip='
            # f' --codec2_port='
        )

        x = config.copy()
        if 'tlm_decode' in x:
            x.pop('tlm_decode')
        if all(map(lambda v: v is None, x.values())):
            if self.sat_name.isnumeric():
                config['norad'] = int(self.sat_name)
            else:
                config['name'] = self.sat_name

        self.sat_fg = sats.SatFlowgraph(self.log, samp_rate, opt_str, is_iq=is_iq, **config)
        if is_iq:
            self.connect(self, self.sat_fg)
        else:
            self.ctor = gr.blocks.complex_to_real()
            self.connect(self, self.ctor, self.sat_fg)

    def start(self):
        utils.unlink(self.tmp_file)

    def finalize(self):
        utils.close(f.f
                    for d in self.sat_fg.get_files().values()
                    for f in d.values())

        d = {}
        for dtype, v in self.sat_fg.get_files().items():
            x = []
            for f in v.values():
                utils.close(f.f)
                x.append(f.path)
            d[dtype] = x
        self.sat_fg.clean()

        self.recorder.satellite.executor.execute(self._sats_finalize, **self.base_kw, files=d)

    @staticmethod
    def _sats_finalize(log: logging.Logger,
                       sat_name: str,
                       files: dict[str, list[pathlib.Path]],
                       observation_key: str,
                       dtype: utils.Decode,
                       **kw) -> tuple[utils.Decode, str, str, dict[str, list[pathlib.Path]]]:

        f_cnt = 0
        for v in files.values():
            f_cnt += len(v)

        log.info('finish %s files', f_cnt)

        return dtype, sat_name, observation_key, files


# class _ProtoMsgDecoder(gr.gr.sync_block):
#     def __init__(self, up: 'ProtoDecoder'):
#         super().__init__(self.__class__.__name__, None, None)
#         self.message_port_register_in(pmt.intern('in'))
#         self.set_msg_handler(pmt.intern('in'), self.handle_msg)
#         self.up = up
#
#     def handle_msg(self, msg):
#         b = pmt.serialize_str(msg)


class ProtoDecoder(Decoder):
    deframer = {
        utils.ProtoDeframer.AALTO1: satellites.components.deframers.aalto1_deframer,
        utils.ProtoDeframer.AAUSAT4: satellites.components.deframers.aausat4_deframer,
        utils.ProtoDeframer.AISTECHSAT_2: satellites.components.deframers.aistechsat_2_deframer,
        utils.ProtoDeframer.AO40_FEC: satellites.components.deframers.ao40_fec_deframer,
        utils.ProtoDeframer.AO40_UNCODED: satellites.components.deframers.ao40_uncoded_deframer,
        utils.ProtoDeframer.ASTROCAST_FX25: satellites.components.deframers.astrocast_fx25_deframer,
        utils.ProtoDeframer.AX100: satellites.components.deframers.ax100_deframer,
        utils.ProtoDeframer.AX25: satellites.components.deframers.ax25_deframer,
        utils.ProtoDeframer.AX5043: satellites.components.deframers.ax5043_deframer,
        utils.ProtoDeframer.BINAR1: satellites.components.deframers.binar1_deframer,
        utils.ProtoDeframer.CCSDS_CONCATENATED: satellites.components.deframers.ccsds_concatenated_deframer,
        utils.ProtoDeframer.CCSDS_RS: satellites.components.deframers.ccsds_rs_deframer,
        utils.ProtoDeframer.DIY1: satellites.components.deframers.diy1_deframer,
        utils.ProtoDeframer.ENDUROSAT: satellites.components.deframers.endurosat_deframer,
        utils.ProtoDeframer.ESEO: satellites.components.deframers.eseo_deframer,
        utils.ProtoDeframer.FOSSASAT: satellites.components.deframers.fossasat_deframer,
        utils.ProtoDeframer.GEOSCAN: satellites.components.deframers.geoscan_deframer,
        utils.ProtoDeframer.GRIZU263A: satellites.components.deframers.grizu263a_deframer,
        utils.ProtoDeframer.HADES: satellites.components.deframers.hades_deframer,
        utils.ProtoDeframer.HSU_SAT1: satellites.components.deframers.hsu_sat1_deframer,
        utils.ProtoDeframer.IDEASSAT: satellites.components.deframers.ideassat_deframer,
        utils.ProtoDeframer.K2SAT: satellites.components.deframers.k2sat_deframer,
        utils.ProtoDeframer.LILACSAT_1: satellites.components.deframers.lilacsat_1_deframer,
        utils.ProtoDeframer.LUCKY7: satellites.components.deframers.lucky7_deframer,
        # utils.ProtoDeframer.MOBITEX: satellites.components.deframers.mobitex_deframer,
        utils.ProtoDeframer.NGHAM: satellites.components.deframers.ngham_deframer,
        utils.ProtoDeframer.NUSAT: satellites.components.deframers.nusat_deframer,
        utils.ProtoDeframer.OPS_SAT: satellites.components.deframers.ops_sat_deframer,
        utils.ProtoDeframer.REAKTOR_HELLO_WORLD: satellites.components.deframers.reaktor_hello_world_deframer,
        utils.ProtoDeframer.SANOSAT: satellites.components.deframers.sanosat_deframer,
        utils.ProtoDeframer.SAT_3CAT_1: satellites.components.deframers.sat_3cat_1_deframer,
        utils.ProtoDeframer.SMOGP_RA: satellites.components.deframers.smogp_ra_deframer,
        utils.ProtoDeframer.SMOGP_SIGNALLING: satellites.components.deframers.smogp_signalling_deframer,
        utils.ProtoDeframer.SNET: satellites.components.deframers.snet_deframer,
        utils.ProtoDeframer.SPINO: satellites.components.deframers.spino_deframer,
        utils.ProtoDeframer.SWIATOWID: satellites.components.deframers.swiatowid_deframer,
        utils.ProtoDeframer.TT64: satellites.components.deframers.tt64_deframer,
        utils.ProtoDeframer.U482C: satellites.components.deframers.u482c_deframer,
        utils.ProtoDeframer.UA01: satellites.components.deframers.ua01_deframer,
        utils.ProtoDeframer.USP: satellites.components.deframers.usp_deframer,
        utils.ProtoDeframer.YUSAT: satellites.components.deframers.yusat_deframer,
    }

    def __init__(self,
                 recorder: 'SatRecorder',
                 channels: List[Union[int, float]]):

        self.deftype = recorder.proto_deframer
        name = self.deftype.name + ' Decoder'

        chn = len(channels)
        super(ProtoDecoder, self).__init__(
            recorder,
            max(channels),
            name,
            utils.Decode.PROTO,
            [
                gr.gr.io_signature(1, 1, gr.gr.sizeof_float)
                if chn == 1
                else gr.gr.io_signature.makev(chn, chn, [gr.gr.sizeof_float] * chn),
                gr.gr.io_signature(0, 0, 0),
            ],
        )

        self.base_kw['channels'] = channels
        self.base_kw['deftype'] = self.deftype

        self.deframers = []
        self.out = satellites.components.datasinks.kiss_file_sink(str(self.tmp_file))
        self.tmp_file.unlink(True)

        defr = self.deframer[self.deftype]
        opts = recorder.proto_options
        for i, baud in enumerate(channels):
            df = defr(**opts)
            self.deframers.append(df)
            self.connect((self, i), df)
            self.msg_connect((df, 'out'), (self.out, 'in'))

    def start(self):
        super().start()
        self.out.filesink.open(str(self.tmp_file))

    def finalize(self):
        self.out.filesink.do_update()
        self.out.filesink.close()
        self.recorder.satellite.executor.execute(self._proto_finalize, **self.base_kw)

    @staticmethod
    def _proto_finalize(log: logging.Logger,
                        sat_name: str,
                        subname: str,
                        out_dir: pathlib.Path,
                        dtype: utils.Decode,
                        tmp_file: pathlib.Path,
                        observation_key: str,
                        deftype: utils.ProtoDeframer,
                        **kw) -> tuple[utils.Decode, utils.ProtoDeframer, str, str, pathlib.Path, dt.datetime]:
        log.debug('finalizing...')

        st = tmp_file.stat()
        d = kw.get('end_time')
        if d:
            d = dt.datetime.fromisoformat(d)
        else:
            d = dt.datetime.fromtimestamp(st.st_mtime, dateutil.tz.tzutc())
        res_fn = tmp_file.rename(out_dir / d.strftime(f'{sat_name}_%Y-%m-%d_%H-%M-%S,%f_{deftype.name}{subname}.kss'))
        log.info('finish: %s (%s)', res_fn, utils.numbi_disp(st.st_size))
        if not st.st_size:
            res_fn.unlink(True)
            return utils.Decode.NONE,

        return dtype, deftype, sat_name, observation_key, res_fn, d


class ProtoRawDecoder(RawDecoder):
    def __init__(self,
                 recorder: 'SatRecorder',
                 samp_rate: Union[int, float],
                 force_nosend_iq=False,
                 iq_in=True):
        super().__init__(recorder, samp_rate, force_nosend_iq, iq_in,
                         name='Proto Raw Decoder', dtype=utils.Decode.PROTO_RAW)

        self.base_kw['channels'] = recorder.channels
        self.base_kw['deftype'] = recorder.proto_deframer
        self.base_kw['proto_mode'] = recorder.proto_mode
        self.base_kw['proto_options'] = recorder.proto_options

    @staticmethod
    def _raw_finalize(**kw):
        dtype, *x = RawDecoder._raw_finalize(**kw)
        if dtype == utils.Decode.NONE:
            return dtype,
        return dtype, kw['proto_mode'], kw['proto_options'], kw['deftype'], kw['channels'], *x


# needed for typehints
# from sats_receiver.gr_modules.modules import SatRecorder
