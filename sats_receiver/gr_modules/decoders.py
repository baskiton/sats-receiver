import datetime as dt
import logging
import math

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.fft
import gnuradio.filter
import gnuradio.gr

from sats_receiver import utils
from sats_receiver.systems import apt


class Decoder(gr.gr.hier_block2):
    def __init__(self, name, samp_rate, out_dir):
        super(Decoder, self).__init__(
            name,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.now = dt.datetime.fromtimestamp(0)
        self.samp_rate = samp_rate
        self.out_dir = out_dir
        self.tmp_file = out_dir / ('_'.join(name.lower().split()) + '.tmp')

    def start(self):
        self.tmp_file = self.out_dir / ('_'.join([self.t.strftime('%Y%m%d%H%M%S'),
                                                  *self.name().lower().split()]) + '.tmp')

    def finalize(self, sat_name, executor):
        pass

    @property
    def t(self):
        t = dt.datetime.now()
        self.now = t
        return t


class RawDecoder(Decoder):
    def __init__(self, samp_rate, out_dir):
        super(RawDecoder, self).__init__('Raw Decoder', samp_rate, out_dir)

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
        self.tmp_file.unlink(True)

        self.connect(self, self.ctf, self.wav_sink)
        self.connect((self.ctf, 1), (self.wav_sink, 1))

    def start(self):
        super(RawDecoder, self).start()

        self.wav_sink.open(str(self.tmp_file))

    def finalize(self, sat_name, executor):
        self.wav_sink.close()
        if self.tmp_file.exists():
            return self.tmp_file.rename(self.out_dir / dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime).strftime('%Y-%m-%d_%H-%M-%S_RAW.wav'))


class AptDecoder(Decoder):
    def __init__(self, samp_rate, out_dir):
        name = 'APT Decoder'
        super(AptDecoder, self).__init__(name, samp_rate, out_dir)

        self.corr_file = out_dir / ('_'.join(name.lower().split()) + '.corr')
        self.peaks_file = out_dir / ('_'.join(name.lower().split()) + '.peaks')

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
        self.out_file_sink.close()
        self.tmp_file.unlink(True)

        self.out_corr_sink = gr.blocks.file_sink(gr.gr.sizeof_float, str(self.corr_file), False)
        self.out_corr_sink.close()
        self.corr_file.unlink(True)

        self.peak_detector = gr.blocks.peak_detector2_fb(
            threshold_factor_rise=7,
            look_ahead=apt.Apt.IMG_WIDTH * apt.Apt.WORK_RATE // apt.Apt.FINAL_RATE,
            alpha=0.001,
        )
        self.out_peaks_sink = gr.blocks.file_sink(gr.gr.sizeof_char, str(self.peaks_file), False)
        self.out_peaks_sink.close()
        self.peaks_file.unlink(True)

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

    def start(self):
        super(AptDecoder, self).start()
        self.corr_file = self.out_dir / ('_'.join([self.now.strftime('%Y%m%d%H%M%S'),
                                                   *self.name().lower().split()]) + '.corr.tmp')
        self.peaks_file = self.out_dir / ('_'.join([self.now.strftime('%Y%m%d%H%M%S'),
                                                    *self.name().lower().split()]) + '.peaks.tmp')

        self.out_file_sink.open(str(self.tmp_file))
        self.out_file_sink.set_unbuffered(False)

        self.out_corr_sink.open(str(self.corr_file))
        self.out_corr_sink.set_unbuffered(False)

        self.out_peaks_sink.open(str(self.peaks_file))
        self.out_peaks_sink.set_unbuffered(False)

    def finalize(self, sat_name, executor):
        self.out_file_sink.close()
        self.out_corr_sink.close()
        self.out_peaks_sink.close()

        for p in self.tmp_file, self.corr_file, self.peaks_file:
            if not p.exists():
                logging.warning('AptDecoder: %s: missing components `%s`', sat_name, p)
                self.tmp_file.unlink(True)
                self.corr_file.unlink(True)
                self.peaks_file.unlink(True)
                return

        executor.execute(self._finalize, self.out_dir, sat_name,
                         self.tmp_file, self.corr_file, self.peaks_file)

    @staticmethod
    def _finalize(out_dir, sat_name, tmp_file, corr_file, peaks_file):
        logging.debug('AptDecoder: %s: finalizing...', sat_name)

        a = apt.Apt(sat_name, tmp_file, corr_file, peaks_file)
        if a.process():
            logging.info('AptDecoder: %s: finish with error', sat_name)
        else:
            res_fn = out_dir / a.end_time.strftime('%Y-%m-%d_%H-%M-%S.apt')
            res_fn.write_bytes(a.data.tobytes())
            logging.info('AptDecoder: %s: finish: %s (%s)',
                         sat_name, res_fn, utils.numdisp(a.data.size * a.data.itemsize))

        tmp_file.unlink(True)
        corr_file.unlink(True)
        peaks_file.unlink(True)


class RawStreamDecoder(Decoder):
    def __init__(self, samp_rate, out_dir, name='RAW Stream Decoder'):
        super(RawStreamDecoder, self).__init__(name, samp_rate, out_dir)

        self.ctf = gr.blocks.complex_to_float(1)
        self.rail = gr.analog.rail_ff(-1, 1)
        self.ftch = gr.blocks.float_to_char(1, 127)
        # self.fts = gr.blocks.float_to_short(1, 32767)
        # self.stch = gr.blocks.short_to_char(1)

        self.out_file_sink = gr.blocks.file_sink(gr.gr.sizeof_char, str(self.tmp_file), False)
        self.out_file_sink.close()
        self.tmp_file.unlink(True)

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

    def finalize(self, sat_name, executor):
        self.out_file_sink.close()
        if self.tmp_file.exists():
            return self.tmp_file.rename(self.out_dir / dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime).strftime('%Y-%m-%d_%H-%M-%S.s'))


class LrptDecoder(RawStreamDecoder):
    def __init__(self, samp_rate, out_dir):
        super(LrptDecoder, self).__init__(samp_rate, out_dir, name='LRPT Decoder')

    def start(self):
        super(LrptDecoder, self).start()

    def finalize(self, sat_name, executor):
        sf = super(LrptDecoder, self).finalize(sat_name, executor)
