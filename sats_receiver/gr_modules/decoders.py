import datetime as dt
import math

import gnuradio as gr
import gnuradio.blocks
import gnuradio.filter
import gnuradio.gr
import gnuradio.fft


class Decoder(gr.gr.hier_block2):
    def __init__(self, name, samp_rate, out_dir):
        super(Decoder, self).__init__(name,
                                      gr.gr.io_signature(1, 1, gr.gr.sizeof_gr_complex),
                                      gr.gr.io_signature(0, 0, 0))

        self.samp_rate = samp_rate
        self.out_dir = out_dir
        self.tmp_file = out_dir / ('_'.join(name.lower().split()) + '.tmp')

    def start(self):
        pass

    def finalize(self):
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
            gr.blocks.FORMAT_PCM_16,
            False
        )
        self.wav_sink.close()
        self.tmp_file.unlink(True)

        self.connect(self, self.ctf, self.wav_sink)
        self.connect((self.ctf, 1), (self.wav_sink, 1))

    def start(self):
        self.wav_sink.open(str(self.tmp_file))

    def finalize(self):
        self.wav_sink.close()
        if self.tmp_file.exists():
            return self.tmp_file.rename(self.out_dir / dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime).strftime('%Y-%m-%d_%H-%M-%S_RAW.wav'))


class AptDecoder(Decoder):
    APT_IMG_WIDTH = 2080

    def __init__(self, samp_rate, out_dir):
        super(AptDecoder, self).__init__('APT Decoder', samp_rate, out_dir)

        self.work_rate = work_rate = self.APT_IMG_WIDTH * 2 * 4
        carrier_freq = 2400
        resamp_gcd = math.gcd(samp_rate, work_rate)

        self.frs = gr.blocks.rotator_cc(2 * math.pi * -carrier_freq / samp_rate)
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
        self.out_file_sink = gr.blocks.file_sink(gr.gr.sizeof_float, str(self.tmp_file), False)
        self.out_file_sink.close()
        self.tmp_file.unlink(True)

        self.connect(self, self.frs, self.rsp, self.lpf, self.ctm, self.out_file_sink)

    def start(self):
        self.out_file_sink.open(str(self.tmp_file))
        self.out_file_sink.set_unbuffered(False)

    def finalize(self):
        self.out_file_sink.close()
        if self.tmp_file.exists():
            return self.tmp_file.rename(self.out_dir / dt.datetime.fromtimestamp(self.tmp_file.stat().st_mtime).strftime('%Y-%m-%d_%H-%M-%S.apt'))
