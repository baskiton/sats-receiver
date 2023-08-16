import colorsys
import datetime as dt
import enum
import logging
import math
import os
import pathlib

from typing import Union

import dateutil.tz
import ephem
import numpy as np
import scipy as sp
import scipy.signal

from PIL import Image, ImageDraw

from sats_receiver import utils, SYSRESDIR


APTRESDIR = SYSRESDIR / 'NOAA-APT'


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

    def wedge_value(self, num: AptWedgeNum, chan: AptChannel = None) -> float:
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
    FRAME_WIDTH = 2080
    PIX_WIDTH = 4
    FINAL_RATE = 4160
    WORK_RATE = FINAL_RATE * PIX_WIDTH
    SAMPLES_PER_WORK_ROW = FRAME_WIDTH * PIX_WIDTH
    LINE_DUR = dt.timedelta(milliseconds=500)

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
    FRAME_HEIGHT_HF = FRAME_HEIGHT // 2

    SYNC_A = np.array([*map(float, '000011001100110011001100110011000000000')],
                      dtype=np.float32).repeat(PIX_WIDTH) * 2 - 1
    SYNC_B = np.array([*map(float, '000011100111001110011100111001110011100')],
                      dtype=np.float32).repeat(PIX_WIDTH) * 2 - 1
    WEDGE_SAMPLE = np.array([*np.interp([31, 63, 95, 127, 159, 191, 223, 255, 0], [0, 255], [-1, 1])], dtype=np.float32).repeat(BLOCK_HEIGHT)

    # TODO: anoter values???
    # The AVHRR/3 scans 55.4 deg per scan line on either side of the orbital track
    # https://en.wikipedia.org/wiki/NOAA-15#Advanced_Very_High_Resolution_Radiometer_(AVHRR/3)
    #
    # Scanning is cross-track with a range of ±55.37 deg about nadir.
    # https://nwp-saf.eumetsat.int/site/software/aapp/aapp-overview/avhrr-3/
    # https://www.eumetsat.int/avhrr#nav-two
    NOAA_AVHRR_FOV_HALF = 55.37
    NOAA_AVHRR_FOV = NOAA_AVHRR_FOV_HALF * 2

    @classmethod
    def from_apt(cls, aptf: pathlib.Path) -> 'Apt':
        """
        APT file format:
            * 3 lines of TLE data
            * observer lonlat, 2 x double, degrees
            * end time UTC timestamp, double
            * data
        """

        with aptf.open('rb') as f:
            sat_name = f.readline().decode('ascii').rstrip()
            l1 = f.readline().decode('ascii').rstrip()
            l2 = f.readline().decode('ascii').rstrip()
            sat_tle = sat_name, l1, l2
            observer_lonlat = tuple(np.frombuffer(f.read(np.dtype(np.double).itemsize * 2), dtype=np.double))
            end_time = np.frombuffer(f.read(np.dtype(np.double).itemsize), dtype=np.double)[0]
            data = np.fromfile(f, dtype=np.float32)

        x = cls(sat_name, aptf, None, None, sat_tle, observer_lonlat)
        x.data = np.resize(data, (data.size // cls.FRAME_WIDTH, cls.FRAME_WIDTH))
        x.end_time = dt.datetime.fromtimestamp(end_time, dateutil.tz.tzutc())
        x.synced = True

        return x

    def __init__(self,
                 sat_name: str,
                 data_file: pathlib.Path,
                 corr_file: Union[pathlib.Path, None],
                 peaks_file: Union[pathlib.Path, None],
                 sat_tle: tuple[str, str, str],
                 observer_lonlat: tuple[float, float]):
        self.prefix = f'{self.__class__.__name__}: {sat_name}'
        self.log = logging.getLogger(self.prefix)

        self.sat_name = sat_name
        self.data_file = data_file
        self.corr_file = corr_file
        self.peaks_file = peaks_file
        self.peak_coef = 0.4
        self.dev = 0.02
        self.synced = False
        self.sat_ephem = ephem.readtle(*sat_tle)
        self.sat_tle = sat_tle
        self.observer_lonlat = observer_lonlat

        self.dist_dev = int(self.SAMPLES_PER_WORK_ROW * self.dev)
        self.dist_min = self.SAMPLES_PER_WORK_ROW - self.dist_dev
        self.dist_max = self.SAMPLES_PER_WORK_ROW + self.dist_dev

        t = 0
        if self.data_file and self.data_file.is_file():
            t = self.data_file.stat().st_mtime
        self.end_time = dt.datetime.fromtimestamp(t, dateutil.tz.tzutc())
        self.data = np.empty((0, self.FRAME_WIDTH))
        self.map_overlay = np.empty((0, self.IMAGE_WIDTH, 4))

    def to_apt(self, out_dir: pathlib.Path) -> tuple[pathlib.Path, int]:
        """
        APT file format:
            * 3 lines of TLE data
            * observer lonlat, 2 x double, degrees
            * end time UTC timestamp, double
            * data
        :return: result file path and size
        """

        res_fn = out_dir / self.end_time.strftime(f'{self.sat_name}_%Y-%m-%d_%H-%M-%S,%f.apt')
        sz = 0

        with res_fn.open('wb') as f:
            sz += f.write('\n'.join((*self.sat_tle, '')).encode('ascii'))
            sz += f.write(np.array(self.observer_lonlat, dtype=np.double).tobytes())
            sz += f.write(np.array(self.end_time.timestamp(), dtype=np.double).tobytes())
            sz += f.write(self.data.tobytes())

        os.utime(res_fn, (self.end_time.timestamp(), self.end_time.timestamp()))

        return res_fn, sz

    def process(self):
        """
        Run apr data process
        :return: True if process terminated with error
        """
        self.log.debug('process...')

        if not self.synced:
            try:
                tail_cutted, data, peaks_idx = self._prepare_data()
                if not data.size or peaks_idx.size < 5:
                    raise IndexError
            except IndexError:
                self.log.error('invalid received data')
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
        self.log.debug('syncing...')

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
                    self.log.debug('error on line resample: %s', e)
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
        self.log.debug('read telemetry...')

        if self.data.shape[0] < self.FRAME_HEIGHT:
            self.log.error('recording too short for telemetry decoding')
            return 1
        if self.data.shape[0] < self.FRAME_HEIGHT * 2:
            self.log.warning('reading telemetry on short recording, expect unreliable results')

        tlm_a = self.data[0:, self.TLM_A_START:self.SYNC_B_START]
        tlm_b = self.data[0:, self.TLM_B_START:]

        # search the best template with minimum noise
        mean_a = tlm_a.mean(1)
        mean_b = tlm_b.mean(1)

        variance_a = np.var(tlm_a, 1, dtype=np.float32)
        variance_b = np.var(tlm_b, 1, dtype=np.float32)

        corr_a = np.correlate(mean_a, self.WEDGE_SAMPLE)
        corr_b = np.correlate(mean_b, self.WEDGE_SAMPLE)

        qual_a = np.array([cor / np.std(variance_a[i + 7 * self.BLOCK_HEIGHT:i + self.FRAME_HEIGHT], dtype=np.float32)
                           for i, cor in enumerate(corr_a - np.min(corr_a))
                           if variance_a.size > i + (9 * self.BLOCK_HEIGHT)], dtype=np.float32)
        qual_b = np.array([cor / np.std(variance_b[i + 7 * self.BLOCK_HEIGHT:i + self.FRAME_HEIGHT], dtype=np.float32)
                           for i, cor in enumerate(corr_b - np.min(corr_b))
                           if variance_b.size > i + (9 * self.BLOCK_HEIGHT)], dtype=np.float32)

        best_q_a = np.argmax(qual_a)
        best_q_b = np.argmax(qual_b)
        x_a = best_q_a - self.FRAME_HEIGHT_HF
        if x_a < 0:
            best_qni_a = best_q_a
            x_a = 0
        else:
            best_qni_a = self.FRAME_HEIGHT_HF
        best_cqs_a = corr_a[x_a:best_q_a + self.FRAME_HEIGHT_HF]
        x_b = best_q_b - self.FRAME_HEIGHT_HF
        if x_b < 0:
            best_qni_b = best_q_b
            x_b = 0
        else:
            best_qni_b = self.FRAME_HEIGHT_HF
        best_cqs_b = corr_b[x_b:best_q_b + self.FRAME_HEIGHT_HF]

        if np.max(best_cqs_a) >= np.max(best_cqs_b):
            best = best_q_a - best_qni_a + np.argmax(best_cqs_a)
            mean = mean_a
        else:
            best = best_q_b - best_qni_b + np.argmax(best_cqs_b)
            mean = mean_b

        # form the best telemetry data and create combine contrast values
        tlm = mean[best:best + self.FRAME_HEIGHT]
        tlm = np.resize(tlm, (tlm.size // self.BLOCK_HEIGHT, self.BLOCK_HEIGHT)).mean(1)

        clb = tlm[:9]
        hi, lo = clb[7:]
        rng = hi - lo

        if rng > 0:
            # image correction by contrast values
            self.data = (self.data - lo) / rng
        else:
            self.log.warning('invalid telemetry data, perhaps is too noisy')

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

    def create_maps_overlay(self, shapes: utils.MapShapes):
        self.log.debug('create maps overlay...')

        ss_factor = 4
        self._height = height = self.data.shape[0]
        self._ss_height = height * ss_factor
        self._half_ss_height = self._ss_height / 2
        self._ss_width = self.IMAGE_WIDTH * ss_factor
        self._half_ss_width = self._ss_width / 2
        self._line_width = shapes.line_width * ss_factor

        start_time = self.end_time - self.LINE_DUR * height
        sat_positions = np.empty((height, 2), dtype=float)
        self._x_offsets = np.empty(height, dtype=float)
        x_fovs = np.empty(height, dtype=float)

        a = math.radians(self.NOAA_AVHRR_FOV_HALF)
        b = math.pi / 2 - a
        sa = math.sin(a)
        for i in range(height):
            t = start_time + self.LINE_DUR * i
            self.sat_ephem.compute(t)
            sat_positions[i] = self.sat_ephem.sublong, self.sat_ephem.sublat

            c = b - math.acos(sa * (ephem.earth_radius + self.sat_ephem.elevation) / ephem.earth_radius)
            x_fovs[i] = c * 2 / 909

        self._ref_lonlat = sat_positions[height // 2]
        self._y_res = (ephem.separation(sat_positions[0], sat_positions[-1]) / height) / ss_factor
        # x_res = np.mean(x_fovs)
        self._ref_x_res = x_fovs[height // 2] / ss_factor
        self._ref_az = utils.azimuth(self._ref_lonlat, sat_positions[height // 2 + 1])

        for i in range(height):
            self._x_offsets[i] = self._lonlat_to_rel_px(x_fovs[i], sat_positions[i])[0]

        overlay_img = Image.new('RGBA', (self._ss_width, self._ss_height))
        self._draw = ImageDraw.Draw(overlay_img)

        for points, color in shapes.iter():
            if isinstance(points, dict):
                if points['name'] == 'observer':
                    points['lonlat'] = np.radians(self.observer_lonlat)
                self._draw_fig(points, ss_factor)
            else:
                self._draw_lines(points, color)

        self.map_overlay = np.array(overlay_img.resize((self.IMAGE_WIDTH, height), Image.Resampling.LANCZOS),
                                    dtype=np.uint8)

        del self._height
        del self._ss_height
        del self._half_ss_height
        del self._ss_width
        del self._half_ss_width
        del self._line_width
        del self._x_offsets
        del self._ref_lonlat
        del self._y_res
        del self._ref_x_res
        del self._ref_az
        del self._draw

    def _lonlat_to_rel_px(self, x_res, lonlat):
        ab = utils.azimuth(self._ref_lonlat, lonlat) - self._ref_az
        c = min(max(ephem.separation(self._ref_lonlat, lonlat), -utils.THIRD_PI), utils.THIRD_PI)
        a = math.atan(math.cos(ab) * math.tan(c))
        b = math.asin(math.sin(ab) * math.sin(c))
        x = -b / x_res
        y = a / self._y_res
        return x, y

    def _draw_lines(self, points, color):
        to_draw = [(0, 0)] * len(points)

        for i, pt in enumerate(points):
            x, y = self._lonlat_to_rel_px(self._ref_x_res, pt)
            x -= self._x_offsets[round(np.clip(y, 0, self._height - 1))]
            to_draw[i] = x + self._half_ss_width, y + self._half_ss_height

        self._draw.line(to_draw, fill=color, width=self._line_width)

    def _draw_fig(self, fig: dict, ss_factor):
        x, y = self._lonlat_to_rel_px(self._ref_x_res, fig['lonlat'])
        x -= self._x_offsets[round(np.clip(y, 0, self._height - 1))]
        x += self._half_ss_width
        y += self._half_ss_height

        col = fig['color']
        sz = np.multiply(fig['size'], ss_factor)
        if fig['type'] == '+':
            lwidth, llen = sz
            llen /= 2
            self._draw.line(((x - llen, y), (x + llen, y)), fill=col, width=lwidth)
            self._draw.line(((x, y - llen), (x, y + llen)), fill=col, width=lwidth)

        elif fig['type'] == 'o':
            self._draw.ellipse(((x - sz, y - sz), (x + sz, y + sz)), fill=col)

    def black_overlay(self):
        return self.map_overlay * np.array([0, 0, 0, 1], dtype=np.uint8)

    def create_composites(self, *types):
        composites = {
            'HVC': APTRESDIR / f'hvc-{self.sat_name[-2:]}.png',
            'HVCT': APTRESDIR / f'hvct-{self.sat_name[-2:]}.png',
            'NO': APTRESDIR / f'no-{self.sat_name[-2:]}.png',
            'MSA': APTRESDIR / f'msa-{self.sat_name[-2:]}.png',
            'SEA': APTRESDIR / f'sea-{self.sat_name[-2:]}.png',
            'THRM': APTRESDIR / f'thermal-{self.sat_name[-2:]}.png',
        }

        data = (self.data * 255).clip(0, 255).astype(np.uint8)
        ch_a: np.ndarray = data[:, self.IMAGE_A_START:self.TLM_A_START]
        ch_b: np.ndarray = data[:, self.IMAGE_B_START:self.TLM_B_START]

        results = []
        for t in types:
            if t[:-1] in ('HVC', 'HVCT', 'MSA', 'B') and t[-1] == 'P':
                if t[0] == 'B':
                    img = Image.fromarray(ch_b, 'L').convert('RGB')
                else:
                    lut = np.array(Image.open(composites[t[:-1]]), dtype=float) / 255.0
                    img = Image.fromarray((lut[ch_b, ch_a] * 255).clip(0, 255).astype(np.uint8), 'RGB')

                lut = np.array(Image.open(composites['NO']), dtype=float) / 255.0

                for y, line in enumerate(lut):
                    if abs(155 / 360 - colorsys.rgb_to_hsv(*line[0])[0]) < (8 / 255):
                        break

                img.paste(Image.fromarray((lut[ch_b, ch_a] * 255).clip(0, 255).astype(np.uint8), 'RGB'),
                          mask=Image.fromarray((ch_b > y).astype(np.uint8) * 255, 'L'))

                results.append((t, img))

            else:
                lut = np.array(Image.open(composites[t]), dtype=float) / 255.0
                results.append((t, Image.fromarray((lut[ch_b, ch_a] * 255).clip(0, 255).astype(np.uint8), 'RGB')))

        return results
