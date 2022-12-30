import datetime as dt
import math

import dateutil.tz
import enum
import logging

import numpy as np
import scipy as sp
import scipy.signal


class AptChannel(enum.Enum):
    A = enum.auto()
    B = enum.auto()


class AptWedgeNum(enum.IntEnum):
    ONE = 7
    ZERO = 8
    PRT_1 = 9
    PRT_2 = 10
    PRT_3 = 11
    PRT_4 = 12
    PATCH_TEMP = 13
    BACK_SCAN = 14
    CHAN_IDENT = 15


class AptChannelName(enum.Enum):
    _1 = '1'    # Visible (0.58 – 0.68 μm)
    _2 = '2'    # Near Infrared (0.325 – 1.10 μm)
    _3a = '3a'  # Thermal Infrared (1.58 – 1.64 μm)
    _4 = '4'    # Thermal Infrared (10.30 – 11.30 μm)
    _5 = '5'    # Thermal Infrared (11.50 – 12.50 μm)
    _3b = '3b'  # Thermal Infrared (3.55 – 3.93 μm)


class AptTelemetry:
    CHANNELS_NAMES = [*AptChannelName, None, None, None]

    def __init__(self, ch_a: np.ndarray, ch_b: np.ndarray):
        self.ch_a = ch_a
        self.ch_b = ch_b

        self.channel_a_name = self._channel_name(AptChannel.A)
        self.channel_b_name = self._channel_name(AptChannel.B)

    def wedge_value(self, num: AptWedgeNum, chan: AptChannel = None):
        if chan is None:
            return (self.ch_a[num] + self.ch_b[num]) / 2
        if chan == AptChannel.A:
            return self.ch_a[num]
        if chan == AptChannel.B:
            return self.ch_b[num]

    def _channel_name(self, chan: AptChannel) -> AptChannelName:
        return min(zip(self.CHANNELS_NAMES, self.ch_a[:9]),
                   key=lambda p: abs(p[1] - self.wedge_value(AptWedgeNum.CHAN_IDENT, chan)))[0]


class AptFrame:
    def __init__(self, data: np.ndarray, tlm_a: np.ndarray, tlm_b: np.ndarray):
        self.data = data
        self.tlm = AptTelemetry(tlm_a, tlm_b)

    @property
    def img_a(self) -> np.ndarray:
        return self.data[:, Apt.IMAGE_A_START:Apt.TLM_A_START]

    @property
    def img_b(self) -> np.ndarray:
        return self.data[:, Apt.IMAGE_B_START:Apt.TLM_B_START]

    def set_tlm_calibration(self, c: np.ndarray):
        self.tlm.ch_a[:AptWedgeNum.PRT_1] = c
        self.tlm.ch_b[:AptWedgeNum.PRT_1] = c


class Apt:
    CARRIER_FREQ = 2400
    IMG_WIDTH = 2080
    PIX_WIDTH = 4
    FINAL_RATE = 4160
    WORK_RATE = FINAL_RATE * PIX_WIDTH
    SAMPLES_PER_WORK_ROW = IMG_WIDTH * PIX_WIDTH

    SYNC_WIDTH = 39
    SPACE_WIDTH = 47
    IMAGE_WIDTH = 909
    TLM_WIDTH = 45

    SYNC_A_START = 0
    SPACE_A_START = SYNC_A_START + SYNC_WIDTH
    IMAGE_A_START = SPACE_A_START + SPACE_WIDTH
    TLM_A_START = IMAGE_A_START + IMAGE_WIDTH

    SYNC_B_START = TLM_A_START + TLM_WIDTH
    SPACE_B_START = SYNC_B_START + SYNC_WIDTH
    IMAGE_B_START = SPACE_B_START + SPACE_WIDTH
    TLM_B_START = IMAGE_B_START + IMAGE_WIDTH

    BLOCK_HEIGHT = 8
    BLOCKS_NUM = 16
    FRAME_HEIGHT = BLOCK_HEIGHT * BLOCKS_NUM

    SYNC_A = np.array([*map(float, '000011001100110011001100110011000000000')],
                      dtype=np.float32).repeat(PIX_WIDTH) * 2 - 1
    SYNC_B = np.array([*map(float, '000011100111001110011100111001110011100')],
                      dtype=np.float32).repeat(PIX_WIDTH) * 2 - 1
    WEDGE_SAMPLE = np.array([*np.interp([31, 63, 95, 127, 159, 191, 223, 255, 0], [0, 255], [-1, 1]),
                             0, 0, 0, 0, 0, 0, 0], dtype=np.float32).repeat(BLOCK_HEIGHT)

    @classmethod
    def from_synced_aptfile(cls, sat_name, aptf) -> 'Apt':
        x = cls(sat_name, aptf, None, None)
        x.data = np.fromfile(aptf, dtype=np.float32)
        x.data = np.resize(x.data, (x.data.size // cls.IMG_WIDTH, cls.IMG_WIDTH))
        x.synced = True
        return x

    @classmethod
    def from_synced_apt(cls, sat_name, apt: np.ndarray, end_time: dt.datetime):
        x = cls(sat_name, None, None, None)
        x.data = np.resize(apt, (apt.size // cls.IMG_WIDTH, cls.IMG_WIDTH))
        x.end_time = end_time
        x.synced = True
        return x

    def __init__(self, sat_name, data_file, corr_file, peaks_file):
        self.sat_name = sat_name
        self.data_file = data_file
        self.corr_file = corr_file
        self.peaks_file = peaks_file
        self.peak_coef = 0.4
        self.dev = 0.02
        self.synced = False

        self.dist_dev = int(self.SAMPLES_PER_WORK_ROW * self.dev)
        self.dist_min = self.SAMPLES_PER_WORK_ROW - self.dist_dev
        self.dist_max = self.SAMPLES_PER_WORK_ROW + self.dist_dev

        t = 0
        if self.data_file and self.data_file.is_file():
            t = self.data_file.stat().st_mtime
        self.end_time = dt.datetime.fromtimestamp(t, dateutil.tz.tzlocal())
        self.data = np.empty((0, self.IMG_WIDTH))

    def process(self):
        logging.debug('Apt: %s: process...', self.sat_name)

        if not self.synced:
            try:
                tail_cutted, data, peaks_idx = self._prepare_data()
                if not data.size or peaks_idx.size < 5:
                    raise IndexError
            except IndexError:
                logging.error('Apt: %s: invalid received data', self.sat_name)
                return 1

            self._syncing(tail_cutted, data, peaks_idx)

        return self._read_telemetry()

    def _prepare_data(self) -> tuple[int, np.ndarray, np.ndarray]:
        data: np.ndarray = np.fromfile(self.data_file, dtype=np.float32)
        corrs: np.ndarray = np.fromfile(self.corr_file, dtype=np.float32)
        peaks: np.ndarray = np.fromfile(self.peaks_file, dtype=np.byte)

        x = np.flatnonzero(corrs > (np.max(corrs[np.flatnonzero(peaks)]) * self.peak_coef))
        start_pos, end_pos = x[0], x[-1]
        tail_cutted = data.size - end_pos

        return tail_cutted, data[start_pos:end_pos], np.flatnonzero(peaks[start_pos:end_pos])

    def _syncing(self, tail_cutted, data, peaks_idx):
        logging.debug('Apt: %s: syncing...', self.sat_name)

        peaks = [peaks_idx[0]]
        it = iter(range(1, peaks_idx.size))
        for i in it:
            prev = peaks_idx[i - 1]
            cur = peaks_idx[i]
            d = cur - prev

            if d < self.dist_min or d > self.dist_max:
                i_ = i
                for i_ in it:
                    prev_ = peaks_idx[i_ - 1]
                    cur_ = peaks_idx[i_]
                    k_ = prev_ - prev
                    d_ = cur_ - prev_

                    if self.dist_min < d_ < self.dist_max:
                        for j in range(1, round(k_ / self.SAMPLES_PER_WORK_ROW)):
                            peaks.append(prev + self.SAMPLES_PER_WORK_ROW * j)

                        k_ = prev_ - peaks[-1]
                        if self.dist_min < k_ < self.dist_max or k_ > self.dist_dev:
                            peaks.append(prev_)
                        else:
                            peaks[-1] = prev_

                        peaks.append(cur_)
                        break

                else:
                    prev_ = peaks_idx[i_ - 1]
                    cur_ = peaks_idx[i_]
                    d_ = cur_ - prev_

                    m = round(d_ / self.SAMPLES_PER_WORK_ROW)
                    if self.dist_min * m < d_ < self.dist_max * m:
                        m += 1

                    for j in range(1, m):
                        peaks.append(prev_ + self.SAMPLES_PER_WORK_ROW * j)

            else:
                peaks.append(cur)

        result = np.full((len(peaks), self.SAMPLES_PER_WORK_ROW), np.nan, dtype=np.float32)
        without_last = err = 0

        for idx, i in enumerate(peaks):
            try:
                x = data[i:peaks[idx + 1]]
            except IndexError:
                z = data.size - i
                if z < self.dist_min:
                    without_last = 1
                    break

                x = data[i:i + self.SAMPLES_PER_WORK_ROW]

            try:
                x = sp.signal.resample(x, self.SAMPLES_PER_WORK_ROW)
            except ValueError as e:
                if not err:
                    logging.debug('Apt: %s: error on line resample: %s', self.sat_name, e)
                    err = 1
                continue

            result[idx] = x

        result_end_pos = (np.argmax(np.isnan(result).all(axis=1)) or result.shape[0]) - without_last
        result_tail_cutted = data.size / self.SAMPLES_PER_WORK_ROW - result_end_pos
        tail_cutted /= self.SAMPLES_PER_WORK_ROW

        self.end_time -= dt.timedelta(milliseconds=(tail_cutted + result_tail_cutted) * 500)
        self.data = result[0:result_end_pos, self.PIX_WIDTH // 2::self.PIX_WIDTH]
        self.synced = True

    def _read_telemetry(self):
        logging.debug('Apt: %s: read telemetry...', self.sat_name)

        if self.data.shape[0] < self.WEDGE_SAMPLE.size:
            logging.error('Apt: %s: recording too short for telemetry decoding', self.sat_name)
            return 1
        if self.data.shape[0] < self.WEDGE_SAMPLE.size * 2:
            logging.warning('Apt: %s: reading telemetry on short recording, expect unreliable results', self.sat_name)

        tlm_a = self.data[0:, self.TLM_A_START:self.SYNC_B_START]
        tlm_b = self.data[0:, self.TLM_B_START:]

        # search the best template with minimum noise
        mean_a = np.mean(tlm_a, 1, dtype=np.float32)
        mean_b = np.mean(tlm_b, 1, dtype=np.float32)
        variance = (np.var(tlm_a, 1, dtype=np.float32)
                    + np.var(tlm_b, 1, dtype=np.float32)) / 2
        corr = np.correlate(mean_a, self.WEDGE_SAMPLE) + np.correlate(mean_b, self.WEDGE_SAMPLE)
        qual = np.array([v / np.std(variance[i:i + self.WEDGE_SAMPLE.size], dtype=np.float32)
                         for i, v in enumerate(corr)], dtype=np.float32)
        best = np.argmax(qual)

        # form the best telemetry data and create combine contrast values
        tlm_a = np.reshape(mean_a[best:best + self.FRAME_HEIGHT], (-1, self.BLOCK_HEIGHT)).mean(1, dtype=np.float32)
        tlm_b = np.reshape(mean_b[best:best + self.FRAME_HEIGHT], (-1, self.BLOCK_HEIGHT)).mean(1, dtype=np.float32)
        clb = (tlm_a[:9] + tlm_b[:9]) / 2
        hi, lo = clb[7:10]
        rng = hi - lo

        if rng > 0:
            # image correction by contrast values
            self.data = (self.data - lo) / rng
        else:
            logging.warning('Apt: %s: invalid telemetry data, perhaps is too noisy', self.sat_name)

        # TODO
        # frames_num = math.ceil(mean_a.shape[0] / self.FRAME_HEIGHT)
        # first_frame_height = best % self.FRAME_HEIGHT
        #
        # frames format: start, height, quality
        # frames = [0, first_frame_height, qual[0]]
        # for i in range(first_frame_height, self.data.shape[0], self.FRAME_HEIGHT):
        #     try:
        #         q = qual[i]
        #     except IndexError:
        #         q = 0
        #     frames.append((i, self.FRAME_HEIGHT, q))
        # a, b, c = frames[-1]
        # frames[-1] = a, self.data.shape[0] - a, c
