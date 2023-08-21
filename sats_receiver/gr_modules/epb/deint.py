import io

import gnuradio as gr
import gnuradio.gr
import numpy as np
import pmt
import pmt.pmt_python
import pmt.pmt_to_python
import pmt.pmt_to_python


from sats_receiver import utils


F32_ITEMSIZE = 4
F32_ITEMSIZE_LOG = 2

INTER_MARKER = 0x27
INTER_MARKER_STRIDE = 80
INTER_MARKER_INTERSAMPS = INTER_MARKER_STRIDE - 8
INTER_BRANCH_COUNT = 36
INTER_BRANCH_DELAY = 2048
INTER_BRANCH_DCNT = INTER_BRANCH_DELAY * INTER_BRANCH_COUNT

period = INTER_MARKER_STRIDE // 8
syncwords = np.array((0x27, 0x4E, 0xD8, 0xB1), np.uint8)

CADU_DATA_LENGTH = 1020
CONV_CADU_LEN = 2 * 1024
CADU_SOFT_LEN = 8192
CADU_SOFT_CHUNK = 8192


class Deinterleave(gr.gr.basic_block):
    MPDU_DATA_SIZE = 882
    INSERT_ZONE_SIZE = 2

    def __init__(self, sample_sz=CADU_SOFT_CHUNK):
        super(Deinterleave, self).__init__(
            name='Deinterleave',
            in_sig=[np.float32],
            out_sig=None,
        )

        self.message_port_register_out(pmt.intern('out'))

        self.sample_sz = sample_sz
        self.out_buf = np.empty((2 * CADU_SOFT_LEN) * 10 // 9 + 8, dtype=np.int8)
        self.io = io.BytesIO()
        self._cur_branch = self._offset = self.offset = 0
        self._deint = np.empty(INTER_BRANCH_COUNT * INTER_BRANCH_DCNT, np.int8)

        self.start_dst_off = INTER_MARKER_STRIDE
        self.rotation = utils.Phase.PHASE_0
        self.from_prev = np.empty(INTER_MARKER_STRIDE, np.int8)

    def _num_samples(self):
        x = (int(not self._cur_branch)
             + (self.sample_sz - (INTER_MARKER_INTERSAMPS - self._cur_branch)
                + INTER_MARKER_INTERSAMPS - 1) // INTER_MARKER_INTERSAMPS)
        return self.sample_sz + 8 * x

    def general_work(self, input_items, output_items):
        inp: np.ndarray = input_items[0]

        self.io.seek(0, io.SEEK_END)
        self.io.write(inp.tobytes())
        self.consume(0, inp.size)

        while 1:
            if self.read_samples():
                return 0

            x = self.out_buf[:CADU_SOFT_LEN]
            self.message_port_pub(
                pmt.intern('out'),
                pmt.cons(pmt.PMT_NIL, pmt.init_f32vector(x.size, x)),
            )

    def read_samples(self):
        need = self._num_samples()

        self.io.seek(0, io.SEEK_END)
        if (self.io.tell() >> F32_ITEMSIZE_LOG) < need:
            return 1

        if need > 0:
            self.io.seek(0)
            self.out_buf[:need] = np.frombuffer(self.io.read(need << F32_ITEMSIZE_LOG), dtype=np.float32, count=need)

            tmp = self.io.read()
            self.io.seek(0)
            self.io.truncate(0)
            self.io.write(tmp)

        self._deinterleave(self.out_buf, self.out_buf)

    def _general_work(self, input_items, output_items):
        inp: np.ndarray = input_items[0]

        self.io.seek(0, io.SEEK_END)
        self.io.write(inp.tobytes())
        self.consume(0, inp.size)

        while 1:
            for dst_off in range(self.start_dst_off, INTER_MARKER_STRIDE + CADU_SOFT_LEN, self.sample_sz):
                if self._read_samples(dst_off):
                    self.start_dst_off = dst_off
                    return 0

            self.start_dst_off = INTER_MARKER_STRIDE

            x = self.out_buf[INTER_MARKER_STRIDE:INTER_MARKER_STRIDE + CADU_SOFT_LEN]
            self.message_port_pub(
                pmt.intern('out'),
                pmt.cons(pmt.PMT_NIL, pmt.init_f32vector(x.size, x)),
            )

    def _read_samples(self, dst_off):
        num_samples = self._num_samples()
        need = num_samples - self.offset

        self.io.seek(0, io.SEEK_END)
        if (self.io.tell() >> F32_ITEMSIZE_LOG) < need:
            return 1

        n = min(self.offset, num_samples)
        if self.offset:
            self.out_buf[dst_off:dst_off + n] = self.from_prev[:n]
            nn = self.offset - n
            tmp = np.copy(self.from_prev[self.offset:self.offset + nn])
            self.from_prev[:nn] = tmp[:]

        if need > 0:
            self.io.seek(0)
            self.out_buf[dst_off + self.offset:
                         dst_off + self.offset + need] = np.frombuffer(self.io.read(need << F32_ITEMSIZE_LOG),
                                                                       dtype=np.float32, count=need)

            tmp = self.io.read()
            self.io.seek(0)
            self.io.truncate(0)
            self.io.write(tmp)

        self.offset -= n

        if num_samples < (INTER_MARKER_STRIDE << 3):
            self._rotate_soft(self.out_buf[dst_off:], num_samples)
            self._deinterleave(self.out_buf[dst_off:], self.out_buf[dst_off:])
        else:
            old_off = self.offset
            old_rot = self.rotation

            hard = np.packbits(self.out_buf[dst_off:dst_off + (num_samples & ~0x7)] < 0, bitorder='big')
            self.offset = self._autocorrelate(hard)

            deint_off = self._cur_branch and (INTER_MARKER_INTERSAMPS - self._cur_branch)
            self.offset = (self.offset - deint_off + INTER_MARKER_INTERSAMPS + 1) % INTER_MARKER_STRIDE
            self.offset -= (self.offset > (INTER_MARKER_STRIDE >> 1)) and INTER_MARKER_STRIDE

            if self.offset > 0:
                if self.offset > (self.io.tell() >> F32_ITEMSIZE_LOG):
                    self.offset = old_off
                    self.rotation = old_rot
                    return 1
                self.io.seek(0)
                n = dst_off + num_samples
                self.out_buf[n:n + self.offset] = np.frombuffer(self.io.read(self.offset << F32_ITEMSIZE_LOG),
                                                                dtype=np.float32, count=self.offset)

                tmp = self.io.read()
                self.io.seek(0)
                self.io.truncate(0)
                self.io.write(tmp)

            else:
                n = dst_off + num_samples + self.offset
                self.from_prev[:-self.offset] = self.out_buf[n:n - self.offset]

            self._rotate_soft(self.out_buf[dst_off:], num_samples + self.offset)
            self._deinterleave(self.out_buf[dst_off:], self.out_buf[dst_off + self.offset:])
            self.offset = self.offset < 0 and -self.offset

    def _autocorrelate(self, hard: np.ndarray):
        ones_count = np.zeros(period << 3, dtype=np.uint8)
        average_bit = np.zeros((period << 3) + 8, dtype=np.uint8)
        sz = hard.size - hard.size % period

        for i in range(period):
            x = (i << 3) | 7
            j = sz - period + i - 1
            tmp = hard[j]
            for j in range(j - period, -1, -period):
                xor = hard[j] ^ tmp
                tmp, hard[j] = hard[j], xor
                for k in range(8):
                    average_bit[x - k] += (tmp & (1 << k)) and 1 or -1

        window = 0
        h_i = -1
        for i in range((sz - period) << 3):
            x = i % 8
            if not x:
                h_i += 1
            window = (window >> 1) | ((hard[h_i] << x) & 0x80)
            ones_count[i % ones_count.size] += window.bit_count()

        best_idx = 0
        best_corr = ones_count[0] - sz // 64
        i = np.argmin(ones_count)
        if ones_count[i] < best_corr:
            best_idx = int(i)

        tmp = 0
        for i in range(8):
            tmp |= (average_bit[best_idx + i] > 0) and (1 << i)

        self.rotation = utils.Phase(0)
        best_corr = (tmp ^ syncwords[0]).bit_count()
        for i in range(1, syncwords.size):
            corr = (tmp ^ syncwords[i]).bit_count()
            if best_corr > corr:
                best_corr = corr
                self.rotation = utils.Phase(i)

        return best_idx

    def _rotate_soft(self, soft, sz):
        sz &= ~0x1

        if self.rotation == utils.Phase.PHASE_0:
            pass

        elif self.rotation == utils.Phase.PHASE_90:
            x = soft[:sz].reshape((-1, 2))
            soft[:sz] = np.column_stack((x[:, 1], np.negative(x[:, 0]))).flatten()

        elif self.rotation == utils.Phase.PHASE_180:
            np.negative(soft[:sz], out=soft[:sz])

        elif self.rotation == utils.Phase.PHASE_270:
            x = soft[:sz].reshape((-1, 2))
            soft[:sz] = np.column_stack((np.negative(x[:, 1]), x[:, 0])).flatten()

    def _deinterleave(self, dst, src):
        src_i = 0
        read_idx = (self._offset + INTER_BRANCH_DCNT) % self._deint.size

        for _ in range(self.sample_sz):
            if not self._cur_branch:
                src_i += 8
            delay = (self._cur_branch % INTER_BRANCH_COUNT) * INTER_BRANCH_DCNT
            write_idx = (self._offset + self._deint.size - delay) % self._deint.size

            self._deint[write_idx] = src[src_i]
            src_i += 1

            self._offset = (self._offset + 1) % self._deint.size
            self._cur_branch = (self._cur_branch + 1) % INTER_MARKER_INTERSAMPS

        for i in range(self.sample_sz):
            dst[i] = self._deint[read_idx]
            read_idx = (read_idx + 1) % self._deint.size
