import argparse
import datetime as dt
import logging
import pathlib
import shlex

from hashlib import sha256

import construct
import gnuradio as gr
import gnuradio.gr
import pmt

from satellites import filereceiver as grs_filereceivers
from satellites.components import datasinks as grs_datasinks
from satellites.components.datasinks.file_receiver import file_receiver as GrsFileReceiver
from satellites.core import gr_satellites_flowgraph as grs_flowgraph
from satellites.filereceiver.filereceiver import File as GrsFile
from satellites import telemetry as grs_tlm

from sats_receiver import utils
from sats_receiver.systems.satellites import telemetry as sr_tlm
from sats_receiver.systems.satellites import filereceivers as sr_filereceivers
from sats_receiver.systems.satellites import demodulators as sr_demod


class SatFlowgraph(grs_flowgraph):
    @classmethod
    def add_options(cls, parser, file=None, name=None, norad=None):
        super().add_options(parser, file, name, norad)

        data_options = parser.add_argument_group('data sink')
        TlmDecoder.add_options(data_options)
        for i in dir(grs_datasinks):
            if not (i.startswith('__') or i.endswith('__')):
                ds = getattr(grs_datasinks, i)
                if hasattr(ds, 'add_options'):
                    ds.add_options(data_options)

        p_output = parser.add_argument_group('output')
        p_output.add_argument('--hexdump', action='store_true')

    def __init__(self, log, samp_rate=None, options=None,
                 file=None, name=None, norad=None, tlm_decode=False, is_iq=True):
        self.log = log
        self.tlm_decode = tlm_decode
        self._demodulator_hooks['GFSK'] = sr_demod.GfskDemod
        self._demodulator_hooks['GMSK'] = sr_demod.GmskDemod

        if type(options) is str:
            p = argparse.ArgumentParser(prog=self.__class__.__name__,
                                        conflict_handler='resolve')
            self.add_options(p, file, name, norad)
            options = p.parse_args(shlex.split(options))

        super().__init__(file=file, name=name, norad=norad,
                         samp_rate=samp_rate, options=options,
                         iq=is_iq, grc_block=0)

    def _init_datasink(self, key, info):
        if 'decoder' in info:
            ds = getattr(grs_datasinks, info['decoder'])
            try:
                x = ds(options=self.options)
            except TypeError:  # raised if ds doesn't have an options parameter
                x = ds()
        elif 'telemetry' in info:
            x = TlmDecoder(info['telemetry'], self.log, options=self.options, tlm_decode=self.tlm_decode)
        elif 'files' in info:
            x = FileReceiver(info['files'], verbose=False, options=self.options)
        elif 'image' in info:
            x = FileReceiver(info['image'], verbose=False, display=False, fullscreen=False, options=self.options)
        else:
            x = TlmDecoder('raw', self.log, options=self.options)

        self._datasinks[key] = x

    def _init_additional_datasinks(self):
        pass

    def get_files(self) -> dict[str, dict[str, GrsFile]]:
        return {k: v.get_files()
                for k, v in self._datasinks.items()}

    def clean(self):
        for ds in self._datasinks.values():
            ds.clean()


class TlmDecoder(gr.gr.basic_block):
    def __init__(self, dname: str, log: logging.Logger, options=None, tlm_decode=False):
        super().__init__(
            'raw_receiver',
            in_sig=[],
            out_sig=[])
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle_msg)

        self.fmt = getattr(grs_tlm, dname, None)
        if self.fmt is None and dname != 'raw':
            self.fmt = getattr(sr_tlm, dname)

        self.log = log
        self.dname = dname
        self.tlm_decode = tlm_decode
        self.out_dir = pathlib.Path(options.file_output_path)
        self._files: dict[str, GrsFile] = {}

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

        fn_base = '_'.join((self.dname, transmitter, dt.datetime.now().isoformat(),
                            sha256(packet).hexdigest()[:16])).replace(' ', '-').replace('.', ',')

        try:
            tlm = self.fmt.parse(packet)
        except construct.ConstructError as e:
            self.log.debug('TlmDecoder: Could not parse telemetry beacon: %s', e)
            return

        try:
            if not (tlm and self.fmt.build(tlm)):
                return
        except:
            return

        f = GrsFile((self.out_dir / fn_base).with_suffix('.bin'))
        f.f.write(packet)
        to_close = [f.f]
        f.size = len(packet)
        self._files[f.path.name] = f

        if self.tlm_decode:
            tlmf = GrsFile((self.out_dir / fn_base).with_suffix('.txt'))
            tlm = str(tlm)
            tlmf.f.write(tlm.encode('utf-8'))
            to_close.append(tlmf.f)
            tlmf.size = len(tlm)
            self._files[tlmf.path.name] = tlmf

        utils.close(*to_close)

    @classmethod
    def add_options(cls, parser):
        parser.add_argument('--file_output_path', default='.')

    def get_files(self):
        return self._files

    def clean(self):
        self._files = {}


class FileReceiver(GrsFileReceiver):
    def __init__(self, receiver, path=None, verbose=None,
                 options=None, **kwargs):
        gr.gr.basic_block.__init__(
            self,
            'file_receiver',
            in_sig=[],
            out_sig=[])
        if verbose is None:
            if options is not None:
                verbose = options.verbose_file_receiver
            else:
                raise ValueError(
                    'Must indicate verbose in function arguments or options')
        if path is None:
            if options is not None:
                path = options.file_output_path
            else:
                raise ValueError(
                    'Must indicate path in function arguments or options')
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle_msg)

        x = getattr(grs_filereceivers, receiver, None)
        if x is None:
            x = getattr(sr_filereceivers, receiver)
        self.receiver = x(path, verbose, **kwargs)

    def get_files(self):
        return self.receiver._files

    def clean(self):
        self.receiver._files = {}
