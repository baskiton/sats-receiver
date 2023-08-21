import pathlib

import gnuradio as gr
import gnuradio.gr
import numpy as np
import pmt
import pmt.pmt_python
import pmt.pmt_to_python
import pmt.pmt_to_python


class PduToCadu(gr.gr.sync_block):
    MPDU_DATA_SIZE = 882
    INSERT_ZONE_SIZE = 2

    def __init__(self,
                 syncword=b'\x1A\xCF\xFC\x1D',
                 framesize=1024):
        super(PduToCadu, self).__init__(
            name='PDU to CADU',
            in_sig=None,
            out_sig=None,
        )
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle_msg)
        self.out_f = None
        self.syncword = syncword
        self.framesize = framesize

    def set_out_f(self, out_fp: pathlib.Path):
        if self.out_f:
            self.out_f.close()
        self.out_f = out_fp.open('wb')

    def handle_msg(self, msg):
        msg = pmt.cdr(msg)
        if not pmt.is_u8vector(msg):
            # Invalid message type. Expected u8vector
            return

        if not self.out_f:
            # out file not set
            return

        cadu = bytes(pmt.u8vector_elements(msg))

        if self.syncword:
            self.out_f.write(self.syncword)
        self.out_f.write(cadu)
        self.out_f.write(np.zeros(self.framesize - len(cadu) - len(self.syncword), dtype=np.uint8).tobytes())

    def stop(self):
        if self.out_f:
            self.out_f.close()
        return super().stop()
