import datetime as dt
import logging
import math

import gnuradio as gr
import gnuradio.analog
import gnuradio.blocks
import gnuradio.digital
import gnuradio.filter
import gnuradio.gr
import gnuradio.fft

import numpy as np
import scipy as sp
import scipy.signal


class Decoder(gr.gr.hier_block2):
    def __init__(self, name, samp_rate, out_dir):
        super(Decoder, self).__init__(
            name,
            gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
            gr.gr.io_signature(0, 0, 0)
        )

        self.samp_rate = samp_rate
        self.out_dir = out_dir
        self.tmp_file = out_dir / ('_'.join(name.lower().split()) + '.tmp')

    def start(self):
        self.tmp_file = self.out_dir / ('_'.join([dt.datetime.now().strftime('%Y%m%d%H%M%S'),
                                                  *self.name().lower().split()]) + '.tmp')

    def finalize(self, sat_name):
        pass


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

    def finalize(self, sat_name):
        self.wav_sink.close()
        if self.tmp_file.exists():
            return self.tmp_file.rename(self.out_dir / dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime).strftime('%Y-%m-%d_%H-%M-%S_RAW.wav'))


class AptDecoder(Decoder):
    APT_CARRIER_FREQ = 2400
    APT_FINAL_RATE = 4160
    APT_IMG_WIDTH = 2080
    APT_SYNC_A = '000011001100110011001100110011000000000'
    APT_SYNC_B = '000011100111001110011100111001110011100'

    def __init__(self, samp_rate, out_dir):
        name = 'APT Decoder'
        super(AptDecoder, self).__init__(name, samp_rate, out_dir)

        self.corr_file = out_dir / ('_'.join(name.lower().split()) + '.corr')
        self.peaks_file = out_dir / ('_'.join(name.lower().split()) + '.peaks')

        self.work_rate = work_rate = self.APT_IMG_WIDTH * 2 * 4
        self.samples_per_work_row = self.APT_IMG_WIDTH * work_rate // self.APT_FINAL_RATE
        self.decim_factor = self.work_rate // self.APT_FINAL_RATE
        self.dev = 0.02
        self.dist_dev = int(self.samples_per_work_row * self.dev)
        self.dist_min = self.samples_per_work_row - self.dist_dev
        self.dist_max = self.samples_per_work_row + self.dist_dev

        resamp_gcd = math.gcd(samp_rate, work_rate)
        pix_width = work_rate // self.APT_FINAL_RATE
        sync_a, sync_b = self._generate_sync_frame(samp_rate, self.APT_FINAL_RATE)

        self.frs = gr.blocks.rotator_cc(2 * math.pi * -self.APT_CARRIER_FREQ / samp_rate)
        self.rsp = gr.filter.rational_resampler_ccc(
            interpolation=work_rate // resamp_gcd,
            decimation=samp_rate // resamp_gcd,
            taps=[],
            fractional_bw=0
        )

        tofilter_freq = 2080
        resamp_gcd = math.gcd(work_rate, tofilter_freq)

        self.lpf = gr.filter.fir_filter_ccf(
            1,
            gr.filter.firdes.low_pass(
                1,
                work_rate / resamp_gcd,
                tofilter_freq / resamp_gcd,
                0.1,
                gr.fft.window.WIN_HAMMING,
                6.76
            )
        )
        self.ctm = gr.blocks.complex_to_mag()
        self.ftc = gr.blocks.float_to_complex()
        # self.dc_rem = gr.blocks.correctiq()
        self.correllator = gr.digital.corr_est_cc(sync_a, pix_width, 1, 0.9, gr.digital.THRESHOLD_ABSOLUTE)
        self.ctf_out = gr.blocks.complex_to_float()
        self.ctf_corr = gr.blocks.complex_to_float()

        self.out_file_sink = gr.blocks.file_sink(gr.gr.sizeof_float, str(self.tmp_file), False)
        self.out_file_sink.close()
        self.tmp_file.unlink(True)

        self.out_corr_sink = gr.blocks.file_sink(gr.gr.sizeof_float, str(self.corr_file), False)
        self.out_corr_sink.close()
        self.corr_file.unlink(True)

        self.peak_detector = gr.blocks.peak_detector2_fb(7, self.samples_per_work_row, 0.001)
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
        self.corr_file = self.out_dir / ('_'.join([dt.datetime.now().strftime('%Y%m%d%H%M%S'),
                                                   *self.name().lower().split()]) + '.corr.tmp')
        self.peaks_file = self.out_dir / ('_'.join([dt.datetime.now().strftime('%Y%m%d%H%M%S'),
                                                    *self.name().lower().split()]) + '.peaks.tmp')

        self.out_file_sink.open(str(self.tmp_file))
        self.out_file_sink.set_unbuffered(False)

        self.out_corr_sink.open(str(self.corr_file))
        self.out_corr_sink.set_unbuffered(False)

        self.out_peaks_sink.open(str(self.peaks_file))
        self.out_peaks_sink.set_unbuffered(False)

    def finalize(self, sat_name):
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

        start_pos, end_pos, data, peaks_idx = self._prepare_data(self.tmp_file, self.corr_file, self.peaks_file)
        tail_correct = end_pos
        peaks = [peaks_idx[0]]

        it = iter(range(1, peaks_idx.size))
        for i in it:
            prev = peaks_idx[i - 1]
            cur = peaks_idx[i]
            d = cur - prev

            if d < self.dist_min or d > self.dist_max:
                i_ = i
                for i_ in it:
                    prev_ = peaks_idx[i_ - 1]
                    cur_ = peaks_idx[i_]

                    if self.dist_min < (cur_ - prev_) < self.dist_max:
                        for j in range(1, i_ - i):
                            peaks.append(prev + self.samples_per_work_row * j)
                        peaks.append(prev_)
                        peaks.append(cur_)
                        break

                else:
                    prev_ = peaks_idx[i_ - 1]
                    cur_ = peaks_idx[i_]
                    d_ = cur_ - prev_

                    m = math.ceil(d_ / self.samples_per_work_row)
                    if self.dist_min * m < d_ < self.dist_max * m:
                        m += 1

                    for j in range(1, m):
                        peaks.append(prev_ + self.samples_per_work_row * j)

            else:
                peaks.append(cur)

        result = np.zeros((len(peaks), self.samples_per_work_row), dtype=np.float32)
        without_last = err = 0

        for idx, i in enumerate(peaks):
            try:
                x = data[i:peaks[idx + 1]]
            except IndexError:
                x = data[i:]
                if x.size < self.dist_min:
                    tail_correct += x.size
                    without_last = 1
                    break
            try:
                x = sp.signal.resample(x, self.samples_per_work_row)
            except ValueError as e:
                if not err:
                    logging.debug('AptDecoder: %s: error on line resample: %s', sat_name, e)
                    err = 1
                continue
            result[idx] = x

        result = result[0:-1 - without_last, self.decim_factor // 2::self.decim_factor]
        # result = (result * 255).round().clip(0, 255).astype(np.uint8)

        tail_correct /= self.work_rate
        end_time = dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime - tail_correct)

        self.tmp_file.write_bytes(result.tobytes())
        res_fn = self.tmp_file.rename(self.out_dir / end_time.strftime('%Y-%m-%d_%H-%M-%S.apt'))

        self.corr_file.unlink(True)
        self.peaks_file.unlink(True)

        return res_fn

    def _generate_sync_frame(self, work_rate, final_rate) -> tuple[np.array, np.array]:
        if work_rate % final_rate:
            raise ValueError('work_rate is not multiple of final_rate')

        pix_width = work_rate // final_rate

        sync_a = np.array([*map(float, self.APT_SYNC_A)], dtype=np.int8).repeat(pix_width)
        sync_a[sync_a == 0] = -1
        sync_b = np.array([*map(float, self.APT_SYNC_B)], dtype=np.int8).repeat(pix_width)
        sync_b[sync_b == 0] = -1

        return sync_a.astype(np.float32), sync_b.astype(np.float32)

    def _prepare_data(self, dataf, corrf, peaksf) -> tuple[int, int, np.ndarray, np.ndarray]:
        data: np.ndarray = np.fromfile(dataf, dtype=np.float32)
        corrs: np.ndarray = np.fromfile(corrf, dtype=np.float32)
        peaks: np.ndarray = np.fromfile(peaksf, dtype=np.byte)

        x = np.flatnonzero(corrs > (np.max(corrs[np.flatnonzero(peaks)]) * 0.4))
        start_pos, end_pos = x[0], x[-1]

        return start_pos, end_pos, data[start_pos:end_pos], np.flatnonzero(peaks[start_pos:end_pos])


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

    def finalize(self, sat_name):
        self.out_file_sink.close()
        if self.tmp_file.exists():
            return self.tmp_file.rename(self.out_dir / dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime).strftime('%Y-%m-%d_%H-%M-%S.s'))


class LrptDecoder(RawStreamDecoder):
    def __init__(self, samp_rate, out_dir):
        super(LrptDecoder, self).__init__(samp_rate, out_dir, name='LRPT Decoder')

    def start(self):
        super(LrptDecoder, self).start()

    def finalize(self, sat_name):
        sf = super(LrptDecoder, self).finalize(sat_name)
