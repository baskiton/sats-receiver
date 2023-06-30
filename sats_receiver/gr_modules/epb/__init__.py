import numpy as np

import gnuradio as gr
import gnuradio.gr


class DelayOneImag(gr.gr.sync_block):
    def __init__(self):
        super().__init__(
            name='Delay One Imag',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.imag_last = np.float32(0.0)

    def work(self, input_items, output_items):
        res = input_items[0][:].view(np.float32).reshape((-1, 2))
        imags = np.roll(res[:,1], 1)
        self.imag_last, imags[0] = imags[0], self.imag_last
        res[:, 1] = imags
        output_items[0][:] = res.flatten().view(np.complex64)

        return len(output_items[0])

    def start(self):
        self.imag_last = np.float32(0.0)
        return super().start()
