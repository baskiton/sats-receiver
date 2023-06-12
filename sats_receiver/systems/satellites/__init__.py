import argparse
import logging
import pathlib
import shlex
import sys

from hashlib import sha256

import construct
import gnuradio as gr
import gnuradio.gr
import pmt
import satellites
import satellites.telemetry

from satellites.components import datasinks
from satellites.core import gr_satellites_flowgraph
from satellites.filereceiver import filereceiver

from sats_receiver import utils
from sats_receiver.systems.satellites.usp_tlm import usp


class SatFlowgraph(gr_satellites_flowgraph):
    @classmethod
    def add_options(cls, parser, file=None, name=None, norad=None):
        super().add_options(parser, file, name, norad)

        data_options = parser.add_argument_group('data sink')
        TlmDecoder.add_options(data_options)
        for i in dir(datasinks):
            if not (i.startswith('__') or i.endswith('__')):
                ds = getattr(datasinks, i)
                if hasattr(ds, 'add_options'):
                    ds.add_options(data_options)

        p_output = parser.add_argument_group('output')
        p_output.add_argument('--hexdump', action='store_true')

    def __init__(self, log, samp_rate=None, options=None,
                 file=None, name=None, norad=None, tlm_decode=False):
        self.log = log
        self.tlm_decode = tlm_decode

        if type(options) is str:
            p = argparse.ArgumentParser(prog=self.__class__.__name__,
                                        conflict_handler='resolve')
            self.add_options(p, file, name, norad)
            options = p.parse_args(shlex.split(options))

        super().__init__(file=file, name=name, norad=norad,
                         samp_rate=samp_rate, options=options,
                         iq=1, grc_block=0)

    def _init_datasink(self, key, info):
        if 'decoder' in info:
            ds = getattr(datasinks, info['decoder'])
            try:
                x = ds(options=self.options)
            except TypeError:  # raised if ds doesn't have an options parameter
                x = ds()
        elif 'telemetry' in info:
            x = TlmDecoder(info['telemetry'], self.log, options=self.options, tlm_decode=self.tlm_decode)
        elif 'files' in info:
            x = datasinks.file_receiver(info['files'], verbose=False, options=self.options)
        elif 'image' in info:
            x = datasinks.file_receiver(info['image'], verbose=False, display=False, fullscreen=False, options=self.options)
        else:
            x = TlmDecoder('raw', self.log, options=self.options)

        self._datasinks[key] = x

    def _init_additional_datasinks(self):
        pass

    def get_files(self) -> dict[str, dict[str, filereceiver.File]]:
        return {k: v._files
                for k, v in self._datasinks.items()}


class TlmDecoder(gr.gr.basic_block):
    def __init__(self, dname: str, log: logging.Logger, options=None, tlm_decode=False):
        super().__init__(
            'raw_receiver',
            in_sig=[],
            out_sig=[])
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle_msg)

        self.fmt = getattr(satellites.telemetry, dname, None)
        if self.fmt is None and dname != 'raw':
            self.fmt = getattr(sys.modules[__name__], dname, None)

        self.log = log
        self.dname = dname
        self.tlm_decode = tlm_decode
        self.out_dir = pathlib.Path(options.file_output_path)
        self._files: dict[str, filereceiver.File] = {}

    def handle_msg(self, msg_pmt):
        msg = pmt.cdr(msg_pmt)
        if not pmt.is_u8vector(msg):
            self.log.debug('TlmDecoder: Received invalid message type. Expected u8vector')
            return

        packet = bytes(pmt.u8vector_elements(msg))
        meta = pmt.car(msg_pmt)
        transmitter = pmt.dict_ref(meta, pmt.intern('transmitter'), pmt.PMT_NIL)
        if pmt.is_symbol(transmitter):
            transmitter = pmt.symbol_to_string(transmitter)
        else:
            transmitter = str(transmitter)

        fn_base = '_'.join((self.dname, transmitter, sha256(packet).hexdigest()[:16])).replace(' ', '-')
        f = filereceiver.File((self.out_dir / fn_base).with_suffix('.bin'))

        if self.tlm_decode:
            try:
                tlm = self.fmt.parse(packet)
            except construct.ConstructError as e:
                self.log.debug('TlmDecoder: Could not parse telemetry beacon: %s', e)
                tlm = None
            except AttributeError:
                tlm = packet.hex()

            if tlm:
                tlmf = filereceiver.File((self.out_dir / fn_base).with_suffix('.txt'))
                tlm = str(tlm)
                tlmf.f.write(tlm.encode('utf-8'))
                utils.close(tlmf.f)
                tlmf.size = len(tlm)
                self._files[tlmf.path.name] = f

        f.f.write(packet)
        utils.close(f.f)
        f.size = len(packet)
        self._files[f.path.name] = f

    @classmethod
    def add_options(cls, parser):
        parser.add_argument('--file_output_path', default='.')
