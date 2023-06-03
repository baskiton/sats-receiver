import datetime as dt
import dateutil.tz
import logging
import math
import pathlib

from typing import Optional,  Union

import numpy as np
import scipy as sp
import scipy.signal

from PIL import Image

from sats_receiver import utils


class Sstv:
    MODE = 'L'

    HSYNC_GAP_MS = 0

    HDR_PIX_S = 0.01
    SYNC_PIX_S = 0.001
    """
    SSTV Header Sync Word
    blocks per 10 ms
    _______   _______
      x30  |1|  x30  |  x30
           |_|       |_______
    """
    HDR_SYNC_WORD = np.array([1] * 30 + [-1] + [1] * 30 + [-1] * 30)

    def __init__(self,
                 sat_name: str,
                 out_dir: pathlib.Path,
                 srate: Union[int, float],
                 do_sync=True):
        self.name = self.__class__.__name__
        self.prefix = f'{self.name}: {sat_name}'
        self.log = logging.getLogger(self.prefix)

        self.sat_name = sat_name
        self.out_dir = out_dir
        self.srate = srate
        self.do_sync = do_sync

        self.line_len_fp = self.LINE_S * srate
        self.line_len = int(self.LINE_S * srate)

        # horizontal sync
        self.sync_pix_width = int(srate * self.SYNC_PIX_S)

        self.img = None
        self.img_data_max_size = int(self.line_len_fp * (self.IMG_H + 1)) * np.float32().itemsize
        self.img_data_file = utils.mktmp2(mode='wb',
                                          buffering=0,
                                          dir=out_dir,
                                          prefix='_'.join(self.name.lower().split()),
                                          suffix='.tmp')
        self.img_data_file.truncate(self.img_data_max_size)
        self.img_data_fp = pathlib.Path(self.img_data_file.name)

    def stop(self):
        utils.close(self.img_data_file)
        self.img_data_file = None

    def feed(self, data: np.ndarray) -> int:
        self.img_data_file.write(data.tobytes())
        if self.img_data_file.tell() >= self.img_data_max_size:
            utils.close(self.img_data_file)
            # self.image_process()
            return 1

    def image_process(self):
        self.log.debug('image process...')

        data = np.fromfile(self.img_data_fp, dtype=np.float32, count=self.img_data_max_size)
        data.resize(self.img_data_max_size, refcheck=False)     # resize in-place with zero-filling

        data = self._sync_process(data) if self.do_sync else self._no_sync_process(data)
        data = (data - SstvRecognizer._1500) / (SstvRecognizer._2300 - SstvRecognizer._1500)
        data = self._image_process(data)

        img = Image.fromarray((data * 255).clip(0, 255).astype(np.uint8), self.MODE)
        if self.MODE != 'RGB':
            img = img.convert('RGB')

        self.log.debug('add EXIF')
        self.img = utils.img_add_exif(
            img,
            d=dt.datetime.fromtimestamp(self.img_data_fp.stat().st_mtime, dateutil.tz.tzutc()),
            comment=self.name,
        )
        utils.unlink(self.img_data_fp)

        self.log.debug('image process done')

    def get_image(self) -> Image.Image:
        if not self.img:
            self.image_process()

        return self.img

    def _no_sync_process(self, data: np.ndarray) -> np.ndarray:
        hsync_len = int(self.HSYNC_MS * self.sync_pix_width)
        return np.resize(data, (self.IMG_H, self.line_len))[:, hsync_len:]

    def _sync_process(self, data: np.ndarray) -> np.ndarray:
        self.log.debug('syncing...')

        hsync_len = int(self.HSYNC_MS * self.sync_pix_width)
        line_len = self.line_len - hsync_len

        sync_word = np.array([-1] * hsync_len)
        corrs = np.correlate(data - np.mean(data), sync_word, 'valid')
        corrs_mean = np.mean(corrs)
        corrs_up = corrs > corrs_mean

        k = 0
        peaks = np.empty(self.IMG_H, int)
        for i in range(peaks.size):
            k += np.argmax(corrs_up[k:k + line_len])
            peaks[i] = np.argmax(corrs[k:k + line_len]) + k + hsync_len
            k += line_len

        img_raw = np.zeros((self.IMG_H, line_len), data.dtype)

        for i, line in enumerate(img_raw):
            line[:] = data[peaks[i]:peaks[i] + line_len]

        return img_raw

    def _image_process(self, data: np.ndarray) -> np.ndarray:
        return data


class _Robot(Sstv):
    MODE = 'YCbCr'
    HSYNC_GAP_MS = 3
    C_GAP_MS = 1.5

    def _420(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, c = np.hsplit(data, [int(self.srate * self.Y_MS / 1000)])
        indices = [int(self.srate * self.CSYNC_MS / 1000)]
        c0_s, c0 = np.hsplit(c[::2], indices)
        c1_s, c1 = np.hsplit(c[1::2], indices)

        cr = c0
        cb = c1
        if np.median(c0_s) > np.median(c1_s):
            cr, cb = cb, cr

        return y, np.repeat(cb, 2, axis=0), np.repeat(cr, 2, axis=0)

    def _422(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = [0]
        for i in self.Y_MS, self.CSYNC_MS, self.C_MS, self.CSYNC_MS:
            indices.append(indices[-1] + i)

        indices = np.array(indices[1:]) / 1000
        y, c0_s, c0, c1_s, c1 = np.hsplit(data, (indices * self.srate).astype(int))
        # csync_gap_len = int(self.srate * self.CSYNC_GAP_MS / 1000)
        # c0_m = np.median(c0_s[:, csync_gap_len:])
        # c1_m = np.median(c1_s[:, csync_gap_len:])
        cr = c0
        cb = c1

        return y, cb, cr

    def _image_process(self, data: np.ndarray) -> np.ndarray:
        y, cb, cr = self._color_method(data)
        c_gap_len = int(self.srate * self.C_GAP_MS / 1000)

        # Image.fromarray((data * 255).clip(0, 255).astype(np.uint8), 'L').save('/home/baskiton/pretest.png')
        # Image.fromarray((y * 255).clip(0, 255).astype(np.uint8), 'L').save('/home/baskiton/pretest_y.png')
        # Image.fromarray((cb * 255).clip(0, 255).astype(np.uint8), 'L').save('/home/baskiton/pretest_cb.png')
        # Image.fromarray((cr * 255).clip(0, 255).astype(np.uint8), 'L').save('/home/baskiton/pretest_cr.png')

        return np.dstack((
            sp.signal.resample(y[:, c_gap_len:], self.IMG_W, axis=1),
            sp.signal.resample(cb[:, c_gap_len:], self.IMG_W, axis=1),
            sp.signal.resample(cr[:, c_gap_len:], self.IMG_W, axis=1)
        ))


class Robot24(_Robot):
    VIS = 0x04

    HSYNC_MS = 12
    CSYNC_MS = 6
    Y_MS = 88
    C_MS = 44
    LINE_S = (HSYNC_MS + Y_MS + CSYNC_MS + C_MS + CSYNC_MS + C_MS) / 1000

    IMG_W = 160
    IMG_H = 120

    _color_method = _Robot._422


class Robot36(_Robot):
    VIS = 0x08

    HSYNC_MS = 10.5
    CSYNC_MS = 4.5
    Y_MS = 90
    C_MS = 45
    LINE_S = (HSYNC_MS + Y_MS + CSYNC_MS + C_MS) / 1000

    IMG_W = 320
    IMG_H = 240

    _color_method = _Robot._420


class Robot72(_Robot):
    VIS = 0x0c

    HSYNC_MS = 12
    CSYNC_MS = 6
    Y_MS = 138
    C_MS = 69
    LINE_S = (HSYNC_MS + Y_MS + CSYNC_MS + C_MS + CSYNC_MS + C_MS) / 1000

    IMG_W = 320
    IMG_H = 240

    _color_method = _Robot._422


class _Martin(Sstv):
    MODE = 'RGB'

    HSYNC_MS = 4.862
    CSYNC_MS = 0.572
    C_MS = CSYNC_MS * 256
    LINE_S = (HSYNC_MS + CSYNC_MS + C_MS + CSYNC_MS + C_MS + CSYNC_MS + C_MS + CSYNC_MS) / 1000

    def _image_process(self, data: np.ndarray) -> np.ndarray:
        indices = [0]
        for i in self.CSYNC_MS, self.C_MS, self.CSYNC_MS, self.C_MS, self.CSYNC_MS, self.C_MS:
            indices.append(indices[-1] + i)

        indices = np.array(indices[1:]) / 1000
        _, g, _, b, _, r, _ = np.hsplit(data, (indices * self.srate).astype(int))

        return np.dstack((
            sp.signal.resample(r, self.IMG_W, axis=1),
            sp.signal.resample(g, self.IMG_W, axis=1),
            sp.signal.resample(b, self.IMG_W, axis=1)
        ))


class MartinM1(_Martin):
    VIS = 0x2c

    IMG_W = 320
    IMG_H = 256


class MartinM2(_Martin):
    VIS = 0x28

    C_MS = _Martin.CSYNC_MS * 128
    LINE_S = (_Martin.HSYNC_MS + _Martin.CSYNC_MS + C_MS + _Martin.CSYNC_MS + C_MS + _Martin.CSYNC_MS + C_MS + _Martin.CSYNC_MS) / 1000

    IMG_W = 160
    IMG_H = 256


class MartinM3(_Martin):
    VIS = 0x24

    IMG_W = 320
    IMG_H = 128


class MartinM4(_Martin):
    VIS = 0x20

    C_MS = _Martin.CSYNC_MS * 128
    LINE_S = (_Martin.HSYNC_MS + _Martin.CSYNC_MS + C_MS + _Martin.CSYNC_MS + C_MS + _Martin.CSYNC_MS + C_MS + _Martin.CSYNC_MS) / 1000

    IMG_W = 160
    IMG_H = 128


class _PD(Sstv):
    MODE = 'YCbCr'

    HSYNC_MS = 20
    HSYNC_GAP_MS = 2.08

    def _image_process(self, data: np.ndarray) -> np.ndarray:
        indices = [0]
        for i in self.HSYNC_GAP_MS, self.C_MS, self.C_MS, self.C_MS, self.C_MS:
            indices.append(indices[-1] + i)

        indices = np.array(indices[1:]) / 1000
        _, y1, cr, cb, y2, _ = np.hsplit(data, (indices * self.srate).astype(int))
        y = np.stack((y1, y2), axis=1).reshape((self.IMG_H * 2, -1))

        return np.dstack((
            sp.signal.resample(y, self.IMG_W, axis=1),
            sp.signal.resample(cb.repeat(2, axis=0), self.IMG_W, axis=1),
            sp.signal.resample(cr.repeat(2, axis=0), self.IMG_W, axis=1)
        ))


class PD50(_PD):
    VIS = 0x5d

    C_MS = 91.52
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 320
    IMG_H = 256 // 2


class PD90(_PD):
    VIS = 0x63

    C_MS = 170.24
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 320
    IMG_H = 256 // 2


class PD120(_PD):
    VIS = 0x5f

    C_MS = 121.6
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 640
    IMG_H = 496 // 2


class PD160(_PD):
    VIS = 0x62

    C_MS = 195.854
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 512
    IMG_H = 400 // 2


class PD180(_PD):
    VIS = 0x60

    C_MS = 183.04
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 640
    IMG_H = 496 // 2


class PD240(_PD):
    VIS = 0x61

    C_MS = 244.48
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 640
    IMG_H = 496 // 2


class PD290(_PD):
    VIS = 0x5e

    C_MS = 228.8
    LINE_S = (_PD.HSYNC_MS + _PD.HSYNC_GAP_MS + C_MS * 4) / 1000

    IMG_W = 800
    IMG_H = 616 // 2


class _Scottie(Sstv):
    MODE = 'RGB'

    HSYNC_MS = 9.0
    CSYNC_MS = 1.5
    C_MS = 138.24
    LINE_S = (CSYNC_MS + C_MS + CSYNC_MS + C_MS + HSYNC_MS + CSYNC_MS + C_MS) / 1000

    IMG_W = 320
    IMG_H = 256

    def _image_process(self, data: np.ndarray) -> np.ndarray:
        indices = [0]
        for i in self.CSYNC_MS, self.C_MS, self.CSYNC_MS, self.C_MS, self.CSYNC_MS:
            indices.append(indices[-1] + i)

        indices = np.array(indices[1:]) / 1000
        _, r, _, g, _, b = np.hsplit(data, (indices * self.srate).astype(int))
        g = np.concatenate((np.zeros((1, g.shape[1]), dtype=g.dtype), g[:-1]), axis=0)
        b = np.concatenate((np.zeros((1, b.shape[1]), dtype=b.dtype), b[:-1]), axis=0)

        return np.dstack((
            sp.signal.resample(r, self.IMG_W, axis=1),
            sp.signal.resample(g, self.IMG_W, axis=1),
            sp.signal.resample(b, self.IMG_W, axis=1)
        ))


class ScottieS1(_Scottie):
    VIS = 0x3c


class ScottieS2(_Scottie):
    VIS = 0x38

    C_MS = 88.064
    LINE_S = (_Scottie.CSYNC_MS + C_MS + _Scottie.CSYNC_MS + C_MS + _Scottie.HSYNC_MS + _Scottie.CSYNC_MS + C_MS) / 1000


class ScottieS3(ScottieS1):
    VIS = 0x34

    IMG_W = 160
    IMG_H = 128


class ScottieS4(ScottieS2):
    VIS = 0x30

    IMG_W = 160
    IMG_H = 128


class ScottieDX(_Scottie):
    VIS = 0x4c

    C_MS = 345.6
    LINE_S = (_Scottie.CSYNC_MS + C_MS + _Scottie.CSYNC_MS + C_MS + _Scottie.HSYNC_MS + _Scottie.CSYNC_MS + C_MS) / 1000


class SstvRecognizer:
    STATUS_OK = 0
    STATUS_CALIB_FAIL = 1
    STATUS_VIS_FAIL = 2
    STATUS_VIS_UNKNOWN = 3
    STATUS_FOUND = 4
    STATUS_DONE = 5

    _STATE_0 = 0
    _STATE_GET_PEAKS = 1
    _STATE_GET_HDR = 2
    _STATE_GET_VIS = 3
    _STATE_GET_LINE = 4

    SIGNAL_FREQ_SHIFT = 2300 - 1100
    SIGNAL_TOLERANCE = 50 / SIGNAL_FREQ_SHIFT

    CALIB_LEADER_S = 0.3
    CALIB_BREAK_S = 0.01
    CALIB_AVG_TOLERANCE = 100 / SIGNAL_FREQ_SHIFT
    VIS_BITS = 10
    VIS_BIT_S = 0.03
    VIS_S = VIS_BITS * VIS_BIT_S

    _1100 = (1100 - 1100) / SIGNAL_FREQ_SHIFT
    _1200 = (1200 - 1100) / SIGNAL_FREQ_SHIFT
    _1300 = (1300 - 1100) / SIGNAL_FREQ_SHIFT
    _1400 = (1400 - 1100) / SIGNAL_FREQ_SHIFT
    _1500 = (1500 - 1100) / SIGNAL_FREQ_SHIFT
    _1900 = (1900 - 1100) / SIGNAL_FREQ_SHIFT
    _2300 = (2300 - 1100) / SIGNAL_FREQ_SHIFT

    CODES = {
        Robot24.VIS: Robot24,
        Robot36.VIS: Robot36,
        Robot72.VIS: Robot72,
        MartinM1.VIS: MartinM1,
        MartinM2.VIS: MartinM2,
        MartinM3.VIS: MartinM3,
        MartinM4.VIS: MartinM4,
        PD50.VIS: PD50,
        PD90.VIS: PD90,
        PD120.VIS: PD120,
        PD160.VIS: PD160,
        PD180.VIS: PD180,
        PD240.VIS: PD240,
        PD290.VIS: PD290,
        ScottieS1.VIS: ScottieS1,
        ScottieS2.VIS: ScottieS2,
        ScottieS3.VIS: ScottieS3,
        ScottieS4.VIS: ScottieS4,
        ScottieDX.VIS: ScottieDX,
    }

    def __init__(self,
                 sat_name: str,
                 out_dir: pathlib.Path,
                 srate: Union[int, float],
                 start_peak: int,
                 do_sync=True):
        self.prefix = f'{self.__class__.__name__}: {sat_name}'
        self.log = logging.getLogger(self.prefix)

        self.sat_name = sat_name
        self.out_dir = out_dir
        self.srate = srate
        self.start_peak = start_peak
        self.do_sync = do_sync
        self.state = self._STATE_GET_PEAKS

        # calibration header setup
        self.calib_leader_len = int(self.CALIB_LEADER_S * srate)
        self.calib_break_len = int(self.CALIB_BREAK_S * srate)
        self.calib_len = self.calib_leader_len * 2 + self.calib_break_len
        self.calib_hdr = np.full(self.calib_len, np.nan, np.float32)
        self.calib_rest_sz = self.calib_len

        # vis code setup
        self.vis_len = int(self.VIS_S * srate)
        self.vis = np.full(self.vis_len, np.nan, np.float32)
        self.vis_rest_sz = self.vis_len
        self.vis_code = 0

        self.sstv = None

    def feed(self, input_data: np.ndarray) -> int:
        data = input_data
        res = self.STATUS_OK

        # just in case
        if not self.state:
            return self.STATUS_DONE

        while data.size:
            if self.state == self._STATE_GET_PEAKS:
                data = data[self.start_peak:]
                self.calib_hdr.fill(np.nan)
                self.calib_rest_sz = self.calib_len
                self.vis.fill(np.nan)
                self.vis_rest_sz = self.vis_len
                self.state = self._STATE_GET_HDR

            elif self.state == self._STATE_GET_HDR:
                if not np.isnan(np.amin(self.calib_hdr)):
                    self.calib_hdr.fill(np.nan)
                    self.calib_rest_sz = self.calib_len
                i = np.argmin(self.calib_hdr)
                x = data[:self.calib_rest_sz]
                data = data[self.calib_rest_sz:]
                self.calib_hdr[i:i + x.size] = x
                self.calib_rest_sz -= x.size

                if not self.calib_rest_sz:
                    # hdr is full. check it
                    leaders = np.append(
                        self.calib_hdr[:self.calib_leader_len],
                        self.calib_hdr[self.calib_leader_len + self.calib_break_len
                                       :self.calib_len - self.calib_break_len]
                    )
                    breaker = self.calib_hdr[self.calib_leader_len:self.calib_leader_len + self.calib_break_len]
                    leaders_avg = np.median(leaders)
                    breaker_avg = np.median(breaker)
                    if (math.fabs(leaders_avg - self._1900) < self.CALIB_AVG_TOLERANCE
                            and math.fabs(breaker_avg - self._1200) < self.CALIB_AVG_TOLERANCE):
                        # print(f'    {leaders_avg=} d={math.fabs(leaders_avg - self._1900)}\n'
                        #       f'    {breaker_avg=} d={math.fabs(breaker_avg - self._1200)}')
                        self.state = self._STATE_GET_VIS
                    else:
                        self.state = self._STATE_0
                        return self.STATUS_CALIB_FAIL

            elif self.state == self._STATE_GET_VIS:
                if not np.isnan(np.amin(self.vis)):
                    self.vis.fill(np.nan)
                    self.vis_rest_sz = self.vis_len
                i = np.argmin(self.vis)
                x = data[:self.vis_rest_sz]
                data = data[self.vis_rest_sz:]
                self.vis[i:i + x.size] = x
                self.vis_rest_sz -= x.size

                if not self.vis_rest_sz:
                    # VIS is full. check it
                    vis = np.median(np.resize(self.vis, (10, self.vis_len // 10)), axis=1)
                    vis_bits = [int(bit < self._1200) for bit in vis[8:0:-1]]

                    code = 0
                    for bit in vis_bits[1:]:
                        code = (code << 1) | bit

                    # check parity
                    if sum(vis_bits[1:]) % 2 != vis_bits[0]:
                        if code:
                            self.log.debug('Parity failed VIS<0x%02x>', code)
                        self.state = self._STATE_0
                        return self.STATUS_VIS_FAIL

                    self.vis_code = code
                    sstv = self.CODES.get(code)
                    if not sstv:
                        if code:
                            self.log.debug('Unknown VIS<0x%02x>', code)
                        self.state = self._STATE_0
                        return self.STATUS_VIS_UNKNOWN

                    self.sstv = sstv(sat_name=self.sat_name,
                                     out_dir=self.out_dir,
                                     srate=self.srate,
                                     do_sync=self.do_sync)
                    self.state = self._STATE_GET_LINE
                    res = self.STATUS_FOUND

            elif self.state == self._STATE_GET_LINE:
                if self.sstv.feed(data):
                    self.stop()
                    return self.STATUS_DONE
                break

        return res

    def get_image(self) -> Optional[Image.Image]:
        if self.sstv:
            return self.sstv.get_image()

    def stop(self):
        self.state = self._STATE_0
        if self.sstv:
            self.sstv.stop()
