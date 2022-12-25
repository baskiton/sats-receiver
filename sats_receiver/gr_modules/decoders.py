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

import numpy as np
import scipy as sp
import scipy.signal

from sats_receiver import utils


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

    def finalize(self, sat_name, executor):
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

    def finalize(self, sat_name, executor):
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

        resamp_gcd = math.gcd(samp_rate, work_rate)
        sync_a, sync_b = self._generate_sync_frame()

        self.frs = gr.blocks.rotator_cc(2 * math.pi * -self.APT_CARRIER_FREQ / samp_rate)
        self.rsp = gr.filter.rational_resampler_ccc(
            interpolation=work_rate // resamp_gcd,
            decimation=samp_rate // resamp_gcd,
            taps=[],
            fractional_bw=0,
        )

        tofilter_freq = 2080
        lp_gcd = math.gcd(work_rate, tofilter_freq)

        self.lpf = gr.filter.fir_filter_ccf(
            1,
            gr.filter.firdes.low_pass(
                gain=1,
                sampling_freq=work_rate / lp_gcd,
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
            symbols=sync_a,
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
            look_ahead=self.APT_IMG_WIDTH * work_rate // self.APT_FINAL_RATE,
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

        executor.execute(self._finalize, sat_name, self.tmp_file, self.corr_file, self.peaks_file,
                         self.out_dir, self.work_rate)

    @staticmethod
    def _finalize(sat_name, tmp_file, corr_file, peaks_file, out_dir, work_rate):
        logging.debug('AptDecoder: %s: finalizing...', sat_name)
        try:
            start_pos, end_pos, data, peaks_idx = AptDecoder._prepare_data(tmp_file, corr_file, peaks_file)
            peaks = [peaks_idx[0]]
            if not data.size or peaks_idx.size < 5:
                raise IndexError
        except IndexError:
            logging.error('AptDecoder: %s: invalid received data', sat_name)
            tmp_file.unlink(True)
            corr_file.unlink(True)
            peaks_file.unlink(True)
            return

        samples_per_work_row = AptDecoder.APT_IMG_WIDTH * work_rate // AptDecoder.APT_FINAL_RATE
        decim_factor = work_rate // AptDecoder.APT_FINAL_RATE
        dev = 0.02
        dist_dev = int(samples_per_work_row * dev)
        dist_min = samples_per_work_row - dist_dev
        dist_max = samples_per_work_row + dist_dev

        logging.debug('AptDecoder: %s: syncing...', sat_name)
        it = iter(range(1, peaks_idx.size))
        for i in it:
            prev = peaks_idx[i - 1]
            cur = peaks_idx[i]
            d = cur - prev

            if d < dist_min or d > dist_max:
                i_ = i
                for i_ in it:
                    prev_ = peaks_idx[i_ - 1]
                    cur_ = peaks_idx[i_]
                    k_ = prev_ - prev
                    d_ = cur_ - prev_

                    if dist_min < d_ < dist_max:
                        for j in range(1, round(k_ / samples_per_work_row)):
                            peaks.append(prev + samples_per_work_row * j)

                        k_ = prev_ - peaks[-1]
                        if dist_min < k_ < dist_max or k_ > dist_dev:
                            peaks.append(prev_)
                        else:
                            peaks[-1] = prev_

                        peaks.append(cur_)
                        break

                else:
                    prev_ = peaks_idx[i_ - 1]
                    cur_ = peaks_idx[i_]
                    d_ = cur_ - prev_

                    m = round(d_ / samples_per_work_row)
                    if dist_min * m < d_ < dist_max * m:
                        m += 1

                    for j in range(1, m):
                        peaks.append(prev_ + samples_per_work_row * j)

            else:
                peaks.append(cur)

        result = np.full((len(peaks), samples_per_work_row), np.nan, dtype=np.float)
        without_last = err = 0
        tail_correct = end_pos

        for idx, i in enumerate(peaks):
            try:
                x = data[i:peaks[idx + 1]]
            except IndexError:
                z = data.size - i
                if z < dist_min:
                    tail_correct += z
                    without_last = 1
                    break

                x = data[i:i + samples_per_work_row]

            try:
                x = sp.signal.resample(x, samples_per_work_row)
            except ValueError as e:
                if not err:
                    logging.debug('AptDecoder: %s: error on line resample: %s', sat_name, e)
                    err = 1
                continue

            result[idx] = x

        z = np.argmax(np.isnan(result).all(axis=1))
        if not z:
            z = result.shape[0]

        result = result[0:z - without_last, decim_factor // 2::decim_factor]

        tail_correct /= work_rate
        end_time = dt.datetime.fromtimestamp(tmp_file.stat().st_mtime - tail_correct)

        tmp_file.write_bytes(result.tobytes())
        res_fn = tmp_file.rename(out_dir / end_time.strftime('%Y-%m-%d_%H-%M-%S.apt'))

        corr_file.unlink(True)
        peaks_file.unlink(True)

        logging.debug('AptDecoder: %s: finish: %s (%s)', sat_name, res_fn, utils.numdisp(res_fn.stat().st_size))

    def _generate_sync_frame(self) -> tuple[np.array, np.array]:
        if self.work_rate % self.APT_FINAL_RATE:
            raise ValueError('work_rate is not multiple of final_rate')

        pix_width = self.work_rate // self.APT_FINAL_RATE

        sync_a = np.array([*map(float, self.APT_SYNC_A)], dtype=np.float32).repeat(pix_width) * 2 - 1
        sync_b = np.array([*map(float, self.APT_SYNC_B)], dtype=np.float32).repeat(pix_width) * 2 - 1

        return sync_a, sync_b

    @staticmethod
    def _prepare_data(dataf, corrf, peaksf) -> tuple[int, int, np.ndarray, np.ndarray]:
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
        sf = super(LrptDecoder, self).finalize(sat_name)
