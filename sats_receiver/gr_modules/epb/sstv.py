import gnuradio as gr
import gnuradio.gr
import numpy as np

from sats_receiver.systems import sstv


class SstvEpb(gr.gr.sync_block):
    OUT_IN = 0
    PEAKS_IN = 1

    def __init__(self, srate=16000, do_sync=True, log=None, sat_name='unknown', out_dir=None):
        super(SstvEpb, self).__init__(
            name='SstvEpb',
            in_sig=[np.float32, np.byte],   # [out, corr]
            out_sig=None,
        )
        self.srate = srate
        self.do_sync = do_sync
        self.log = log
        self.sat_name = sat_name
        self.out_dir = out_dir
        self.sstv = []
        self.sstv_done = []
        self.find_sstv = 1

    def work(self, input_items, output_items):
        if self.find_sstv:
            for peak in np.flatnonzero(input_items[self.PEAKS_IN]):
                self.sstv.append(sstv.SstvRecognizer(self.sat_name, self.out_dir, self.srate, peak, self.do_sync))

        to_delete = []
        for i in range(len(self.sstv)):
            status = self.sstv[i].feed(input_items[self.OUT_IN])

            if status == sstv.SstvRecognizer.STATUS_FOUND:
                self.find_sstv = 0
                x = self.sstv[i]
                self.sstv = [x]
                to_delete = []
                self.log.info('SSTV Found: 0x%02x (%s)', x.sstv.VIS, x.sstv.name)
                break

            elif status == sstv.SstvRecognizer.STATUS_DONE:
                self.log.info('%s: Done', self.sstv[i].sstv.name)
                self.sstv_done.append(self.sstv[i])
                self.sstv = []
                self.find_sstv = 1
                break

            elif status == sstv.SstvRecognizer.STATUS_VIS_UNKNOWN:
                if self.sstv[i].vis_code:
                    self.sstv_done.append(self.sstv[i])
                to_delete.append(i)

            elif status != sstv.SstvRecognizer.STATUS_OK:
                to_delete.append(i)

        for i in sorted(to_delete, reverse=True):
            self.sstv.pop(i)

        return input_items[self.PEAKS_IN].size

    def start(self):
        self.find_sstv = 1
        self.sstv = []
        self.sstv_done = []

        return True

    def stop(self):
        self.find_sstv = 0
        for i in self.sstv:
            i.stop()
            self.sstv_done.append(i)
        self.sstv = []

        return True

    def finalize(self):
        rr = self.sstv_done
        self.sstv_done = []

        return rr
