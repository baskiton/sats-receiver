import time

import numpy as np

import gnuradio as gr
import gnuradio.gr


class Prober(gr.gr.sync_block):
    def __init__(self,
                 measure_s=1.0,  # in seconds
                 ):
        super(Prober, self).__init__(
            name='Prober',
            in_sig=[np.float32],
            out_sig=None
        )
        self.measure_s = measure_s
        self._next_measure = 0
        self.counter = 0
        self._changes = -1
        self._prev_counter = 0

    def work(self, input_items, output_items) -> int:
        self.counter += input_items[0].size
        return input_items[0].size

    def is_runned(self) -> bool:
        return bool(self.counter)

    def changes(self) -> int:
        if self.is_runned() and self.t > self._next_measure:
            self._next_measure = self.now + self.measure_s
            self._changes = self.counter - self._prev_counter
            self._prev_counter = self.counter

        return self._changes

    @property
    def t(self) -> float:
        t = time.monotonic()
        self.now = t
        return t
