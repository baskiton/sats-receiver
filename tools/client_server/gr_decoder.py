import datetime as dt
import logging
import logging.handlers
import multiprocessing as mp
import pathlib
import time

from collections import namedtuple as nt
from hashlib import sha256
from typing import Union

import gnuradio as gr
import gnuradio.blocks
import gnuradio.gr

from sats_receiver import utils
from sats_receiver.gr_modules.decoders import Decoder, ProtoDecoder
from sats_receiver.gr_modules.demodulators import FskDemod
from sats_receiver.gr_modules.epb.prober import Prober


sat_nt = nt('Satellite', 'name output_directory executor')
rec_nt = nt('SatRecorder',
            'satellite subname mode '
            'proto_deframer proto_options deviation_factor'
            )


class _Executor:
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

    def action(self, t=30):
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


class ProtoDecoderTopBlock(gr.gr.top_block):
    def __init__(self,
                 wav_fp: Union[pathlib.Path, str],
                 wav_type: utils.RawFileType,
                 recorder: 'SatRecorder',
                 channels=()):
        self.prefix = self.__class__.__name__
        self.log = logging.getLogger(self.prefix)

        if wav_type not in (utils.RawFileType.IQ, utils.RawFileType.AUDIO):
            raise ValueError(f'Invalid wav type: {wav_type.name}')

        super(ProtoDecoderTopBlock, self).__init__('ProtoDecoderTopBlock', catch_exceptions=False)

        self.wav_src = gr.blocks.wavfile_source(str(wav_fp), False)
        self.prober = Prober()
        self.connect(self.wav_src, self.prober)

        if not channels:
            channels = self.wav_src.sample_rate(),

        iq = wav_type == utils.RawFileType.IQ

        self.last = self.wav_src
        with_demod = 0
        if recorder.mode != utils.Mode.RAW:
            if iq:
                self.ftc = gr.blocks.float_to_complex(1)
                self.last = self.ftc
                for i in range(self.wav_src.channels()):
                    self.connect((self.wav_src, i), (self.ftc, i))

            if recorder.mode in (utils.Mode.FSK, utils.Mode.GFSK, utils.Mode.GMSK):
                self.demod = FskDemod(self.wav_src.sample_rate(), channels, recorder.deviation_factor, iq)
            else:
                raise ValueError(f'Invalid mode {recorder.mode}')

            self.connect(self.last, self.demod)
            self.last = self.demod
            with_demod = 1
            channels = getattr(self.demod, 'channels', getattr(self.demod, 'out_rate', channels))

        self.decoder = ProtoDecoder(recorder, channels)
        for i in range(len(channels)):
            self.connect((self.last, with_demod and i or 0), (self.decoder, i))

    def start(self, max_noutput_items=10000000):
        observation_key = sha256((self.prefix + str(dt.datetime.now())).encode()).hexdigest()
        self.log.info('START')
        self.decoder.set_observation_key(observation_key)
        self.decoder.start()
        super(ProtoDecoderTopBlock, self).start(max_noutput_items)

    def stop(self):
        self.log.info('STOP')
        super(ProtoDecoderTopBlock, self).stop()

        self.decoder.finalize()

    def wait(self):
        super(ProtoDecoderTopBlock, self).wait()


def process(fp: pathlib.Path, params: dict):
    out_dir = fp.parent

    executor=_Executor()
    sat = sat_nt(name=params['sat_name'],
                 output_directory=out_dir,
                 executor=executor,
                 )

    recorder = rec_nt(satellite=sat,
                      subname='',
                      mode=utils.Mode[params['proto_mode']],
                      proto_deframer=utils.ProtoDeframer[params['deframer_type']],
                      proto_options=params['proto_options'],
                      deviation_factor=5,
                      )

    tb = ProtoDecoderTopBlock(fp,
                              utils.RawFileType[params['file_type']],
                              recorder,
                              params['channels'],
                              )
    tb.decoder.base_kw['end_time'] = params['end_time']
    tb.start()

    while tb.prober.changes():
        time.sleep(tb.prober.measure_s)

    tb.stop()
    tb.wait()

    x = executor.action()
    print(x)
