import collections
import datetime as dt
import enum
import errno
import gc
import heapq
import itertools
import logging
import math
import os
import pathlib
import struct
import sys
import tempfile
import threading
import time

from typing import Any, Callable, Iterable, Mapping, Union

import ephem
import gnuradio as gr
import gnuradio.blocks
import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import psutil
import shapefile

from PIL import Image, ImageColor, ImageOps, ExifTags
from scipy import fft

plt.set_loglevel('info')


THIRD_PI = math.pi / 3



class Mode(enum.StrEnum):
    RAW = enum.auto()
    AM = enum.auto()
    FM = enum.auto()
    WFM = enum.auto()
    WFM_STEREO = enum.auto()
    QUAD = enum.auto()
    QUAD2FSK = enum.auto()
    SSTV_QUAD = enum.auto()
    QPSK = enum.auto()
    OQPSK = enum.auto()
    GMSK = enum.auto()
    GFSK = enum.auto()
    FSK = enum.auto()
    # AFSK = enum.auto()
    # BPSK = enum.auto()
    LSB = enum.auto()
    USB = enum.auto()
    DSB = enum.auto()


class Decode(enum.StrEnum):
    NONE = enum.auto()
    RAW = enum.auto()
    CSOFT = enum.auto()
    CCSDSCC = enum.auto()
    APT = enum.auto()
    LRPT = enum.auto()
    SSTV = enum.auto()
    SATS = enum.auto()
    PROTO = enum.auto()
    PROTO_RAW = enum.auto()


class ProtoDeframer(enum.StrEnum):
    AALTO1 = enum.auto()
    AAUSAT4 = enum.auto()
    AISTECHSAT_2 = enum.auto()
    AO40_FEC = enum.auto()
    AO40_UNCODED = enum.auto()
    ASTROCAST_FX25 = enum.auto()
    AX100 = enum.auto()
    AX25 = enum.auto()
    AX5043 = enum.auto()
    BINAR1 = enum.auto()
    CCSDS_CONCATENATED = enum.auto()
    CCSDS_RS = enum.auto()
    DIY1 = enum.auto()
    ENDUROSAT = enum.auto()
    ESEO = enum.auto()
    FOSSASAT = enum.auto()
    GEOSCAN = enum.auto()
    GRIZU263A = enum.auto()
    HADES = enum.auto()
    HSU_SAT1 = enum.auto()
    IDEASSAT = enum.auto()
    K2SAT = enum.auto()
    LILACSAT_1 = enum.auto()
    LUCKY7 = enum.auto()
    # MOBITEX = enum.auto()
    NGHAM = enum.auto()
    NUSAT = enum.auto()
    OPS_SAT = enum.auto()
    REAKTOR_HELLO_WORLD = enum.auto()
    SANOSAT = enum.auto()
    SAT_3CAT_1 = enum.auto()
    SMOGP_RA = enum.auto()
    SMOGP_SIGNALLING = enum.auto()
    SNET = enum.auto()
    SPINO = enum.auto()
    SWIATOWID = enum.auto()
    TT64 = enum.auto()
    U482C = enum.auto()
    UA01 = enum.auto()
    USP = enum.auto()
    YUSAT = enum.auto()


class RawOutFormat(enum.Enum):
    NONE = None
    WAV = gr.blocks.FORMAT_WAV
    WAV64 = gr.blocks.FORMAT_RF64
    OGG = gr.blocks.FORMAT_OGG


class RawOutSubFormat(enum.Enum):
    DOUBLE = gr.blocks.FORMAT_DOUBLE
    FLOAT = gr.blocks.FORMAT_FLOAT
    PCM_16 = gr.blocks.FORMAT_PCM_16
    PCM_24 = gr.blocks.FORMAT_PCM_24
    PCM_32 = gr.blocks.FORMAT_PCM_32
    PCM_U8 = gr.blocks.FORMAT_PCM_U8
    VORBIS = gr.blocks.FORMAT_VORBIS
    OPUS = gr.blocks.FORMAT_OPUS


class RawOutDefaultSub(enum.Enum):
    NONE = None
    WAV = RawOutSubFormat.FLOAT
    WAV64 = RawOutSubFormat.FLOAT
    OGG = RawOutSubFormat.VORBIS


class RawFileType(enum.StrEnum):
    IQ = enum.auto()
    WFC = enum.auto()
    AUDIO = enum.auto()


class SsbMode(enum.StrEnum):
    LSB = enum.auto()
    USB = enum.auto()
    DSB = enum.auto()


class Phase(enum.IntEnum):
    PHASE_0 = 0
    PHASE_90 = 1
    PHASE_180 = 2
    PHASE_270 = 3


Event = collections.namedtuple('Event', 't, prior, seq, fn, a, kw')


class Scheduler:
    """
    The scheduler idea is taken from the python stdlib
    and adapted to my needs
    https://github.com/python/cpython/blob/main/Lib/sched.py
    """

    def __init__(self):
        self._queue = []
        self._lock = threading.RLock()
        self._sequence_generator = itertools.count()

    def plan(self, t: dt.datetime, fn: Callable, *a: Any, prior: int = 0, **kw: Any) -> Event:
        with self._lock:
            event = Event(t, prior, next(self._sequence_generator), fn, a, kw)
            heapq.heappush(self._queue, event)
        return event

    def cancel(self, *evt: Event):
        if not evt:
            return

        with self._lock:
            for e in evt:
                try:
                    self._queue.remove(e)
                    heapq.heapify(self._queue)
                except ValueError:
                    pass

    def clear(self):
        with self._lock:
            self._queue.clear()
            heapq.heapify(self._queue)

    def empty(self) -> bool:
        with self._lock:
            return not self._queue

    def action(self, now=None):
        pop = heapq.heappop
        if now is None:
            now = dt.datetime.now(dt.timezone.utc)

        while True:
            with self._lock:
                if not self._queue:
                    break
                t, prior, seq, fn, a, kw = self._queue[0]
                if t > now:
                    return t - now
                else:
                    pop(self._queue)

            fn(*a, **kw)
            # time.sleep(0)
            now = dt.datetime.now(dt.timezone.utc)


class SysUsage:
    DEFAULT_INTV = 3600

    def __init__(self, ctx: str, intv: Union[int, float] = DEFAULT_INTV):
        self.prefix = f'{self.__class__.__name__}: {ctx}'
        self.log = logging.getLogger(self.prefix)
        self.proc = psutil.Process()

        gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
        self.now = 0
        self.intv = intv
        self.next = self.t + intv
        self.ctx = ctx

    def collect(self):
        if self.t >= self.next:
            self.next = self.now + self.intv
            gc.collect()

            with self.proc.oneshot():
                mi = self.proc.memory_info()
                ct = self.proc.cpu_times()

            self.log.debug('%s rss %s utime %s stime %s',
                           numbi_disp(sum(sys.getsizeof(i) for i in gc.get_objects())),
                           numbi_disp(mi.rss),
                           sec(ct.user),
                           sec(ct.system))

    @property
    def t(self) -> float:
        self.now = time.monotonic()
        return self.now


class MapShapes:
    def __init__(self, config: Mapping):
        self.config = config
        self.shapes = []

        for i, shf, col in sorted(config.get('shapes', []), key=lambda x: x[0]):
            self.shapes.append((shf, self._gen_color(col)))

        for name, v in config.get('points', {}).items():
            if name != 'observer':
                v['lonlat'] = np.radians(v['lonlat'])
            v['name'] = name
            v['color'] = self._gen_color(v['color'])
            if v['type'] == '+':
                assert len(v['size']) == 2
                v['size'] = [*map(int, v['size'])]
            elif v['type'] == 'o':
                v['size'] = int(v['size'])
            else:
                raise ValueError(f'Invalid point type: `{v["type"]}`')

            order = int(v.get('order', len(self.shapes)))
            self.shapes.insert(order, (v, v['color']))

    @property
    def shapes_dir(self) -> pathlib.Path:
        return pathlib.Path(self.config['shapes_dir']).expanduser()

    @property
    def line_width(self) -> int:
        return self.config.get('line_width', 1)

    def iter(self) -> tuple[Union[Iterable, Mapping], tuple]:
        for shf, color in self.shapes:
            if isinstance(shf, Mapping):
                yield shf, color
                continue

            for i in shapefile.Reader(self.shapes_dir / shf).iterShapes():
                pts = np.radians(i.points)
                if len(i.parts) <= 1:
                    yield pts, color
                else:
                    for j, k in itertools.pairwise(i.parts):
                        yield pts[j:k], color

    @staticmethod
    def _gen_color(col) -> Iterable[int]:
        alpha = 255
        if isinstance(col, (tuple, list)):
            if len(col) == 4:
                alpha = col[3]
            col = tuple(col[:3]) + (alpha,)
        elif isinstance(col, str):
            col = ImageColor.getcolor(col, 'RGBA')
        elif isinstance(col, int):
            col = col, alpha
        else:
            raise TypeError('Invalid color value type')

        return col


class WaveFormat(enum.IntEnum):
    PCM = 0x0001
    FLOAT = 0x0003
    EXT = 0xFFFE


class WavFile:
    def __init__(self, fp: pathlib.Path):
        self.chunks = {}

        with fp.open('rb') as f:
            fcc = f.read(4)
            if fcc not in (b'RIFF', b'RF64', b'BF64'):
                raise ValueError(f'Unknown format {fcc!r}')

            fsz = struct.unpack('<i', f.read(4))[0]
            if fsz != -1:
                fsz += 8

            x = f.read(4)
            if x != b'WAVE':
                raise ValueError(f'Unknown type {x!r}')

            self._read_chunks(f)

    def _read_chunks(self, f):
        with_fmt = 0
        while 1:
            n = f.read(4)
            if len(n) < 4:
                raise ValueError(f'Incomplete chunk {n!r}')

            chunk_sz, = struct.unpack('<I', f.read(4))

            if n == b'data':
                if not with_fmt:
                    raise ValueError('No fmt chunk before data')

                bytes_per_sample = self.bytes_per_block // self.channels

                if self.audio_format == WaveFormat.PCM:
                    if 1 <= self.bit_depth <= 8:
                        dtype = 'u1'
                    elif self.bit_depth in (3, 5, 6, 7):
                        dtype = 'V1'
                    elif self.bit_depth <= 64:
                        dtype = f'<i{bytes_per_sample}'
                    else:
                        raise ValueError(f'Unsupported bit depth: {self.bit_depth}')

                elif self.audio_format == WaveFormat.FLOAT:
                    if self.bit_depth in (32, 64):
                        dtype = f'<f{bytes_per_sample}'
                    else:
                        raise ValueError(f'Unsupported bit depth: {self.bit_depth}')

                else:
                    raise ValueError(f'Invalid format {self.audio_format.name}')

                start = f.tell()
                if bytes_per_sample not in (1, 2, 4, 8):
                    raise ValueError(f'mmap not compatible with {bytes_per_sample}-bytes container')
                self.data = np.memmap(f, dtype=dtype, mode='c', offset=start).reshape((-1, self.channels))
                self.duration = len(self.data) / self.samp_rate
                break

            elif n == b'ds64':
                self.chunks[n] = f.read(chunk_sz)

            elif n == b'fmt ':
                if chunk_sz < 16:
                    raise ValueError('FMT chunk is not compliant')

                self.audio_format, self.channels, self.samp_rate, self.bytes_per_sec, self.bytes_per_block, self.bit_depth, = struct.unpack('<HHIIHH', f.read(16))
                self.audio_format = WaveFormat(self.audio_format)
                if self.audio_format == WaveFormat.EXT and chunk_sz >= 18:
                    ext_ch_sz, = struct.unpack('<H', f.read(2))
                    if ext_ch_sz >= 22:
                        ext_data = f.read(22)
                        raw_guid = ext_data[6:6 + 16]
                        if raw_guid.endswith(b'\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71'):
                            self.audio_format = WaveFormat(struct.unpack('<I', raw_guid[:4])[0])
                    else:
                        raise ValueError('FMT EXT chunk is not compliant')

                with_fmt = 1

            else:
                print(f'Unknown chunk {n!r}', file=sys.stderr)
                f.seek(chunk_sz, os.SEEK_CUR)
                if chunk_sz % 2:
                    f.seek(1, os.SEEK_CUR)


class WfMode(enum.StrEnum):
    MEAN = enum.auto()
    MAX_HOLD = enum.auto()
    DECIMATION = enum.auto()


class Waterfall:
    """
    borrowed from gr-satnogs:
        https://gitlab.com/librespacefoundation/satnogs/gr-satnogs

    and satnogs-client:
        https://gitlab.com/librespacefoundation/satnogs/satnogs-client

    """

    FILE_HDR_FMT = struct.Struct('!dIIII')
    OFFSET_IN_STDS = -2
    SCALE_IN_STDS = 8

    @classmethod
    def from_wav(cls, in_fn, fft_size=4096, mode=WfMode.MEAN, end_timestamp=0):
        wav = WavFile(pathlib.Path(in_fn))
        if wav.data.dtype != np.float32:
            wav.data = wav.data[:len(wav.data) & -2].astype(np.float32)

        data = wav.data.view(np.complex64).reshape(wav.data.shape[0])

        rps = 10000
        refresh = int((wav.samp_rate / fft_size) / rps) or 1
        data_dtypes = np.dtype([('tabs', 'int64'), ('spec', 'float32', (fft_size, ))])

        fft_shift = math.ceil(fft_size / 2)
        n_fft = len(data) // fft_size
        dur_per_fft_us = (wav.duration * 1000000) / n_fft
        start_timestamp = end_timestamp and end_timestamp - wav.duration

        if not isinstance(mode, WfMode) and isinstance(mode, str):
            mode = WfMode[mode]
        if mode == WfMode.MEAN:
            compute = cls._compute_mean
        elif mode == WfMode.MAX_HOLD:
            compute = cls._compute_max_hold
        elif mode == WfMode.DECIMATION:
            compute = cls._compute_decimation
        else:
            raise ValueError('Invalid waterfall mode')

        data = compute(data, fft_size, n_fft, fft_shift, refresh, dur_per_fft_us, data_dtypes)
        if not data.size:
            raise ValueError('Empty data array')

        return cls(start_timestamp, wav.samp_rate, fft_size, refresh, n_fft, data)

    @classmethod
    def from_cfile(cls, compressed_wf: pathlib.Path):
        with compressed_wf.open('rb') as f:
            hdr = cls.FILE_HDR_FMT.unpack_from(f.read(cls.FILE_HDR_FMT.size))
            fft_size = hdr[2]
            n_fft = hdr[4]
            tabs = np.fromfile(f, np.int64, n_fft)
            spec = cls.spec_decompress(f, fft_size, n_fft)
            data_dtypes = np.dtype([('tabs', 'int64'), ('spec', 'float32', (fft_size, ))])
            data = np.empty(n_fft, dtype=data_dtypes)
            data['tabs'] = tabs
            data['spec'] = spec
            return cls(*hdr, data)

    def __init__(self, start_timestamp, samp_rate, fft_size, refresh, n_fft, data):
        self.start_timestamp = start_timestamp
        self.samp_rate = samp_rate
        self.fft_size = fft_size
        self.refresh = refresh
        self.n_fft = n_fft
        self.data = data

        nint = self.data['spec'].shape[0]
        self.trel = np.arange(nint) * self.refresh * self.fft_size / float(self.samp_rate)
        self.freq = np.linspace(-0.5 * self.samp_rate,
                                0.5 * self.samp_rate,
                                self.fft_size,
                                endpoint=False)

    def to_cfile(self, out_fp: pathlib.Path):
        with out_fp.open('wb') as f:
            f.write(self.FILE_HDR_FMT.pack(self.start_timestamp, self.samp_rate, self.fft_size, self.refresh, self.n_fft))
            self.data['tabs'].tofile(f)
            for i in self.spec_compress():
                i.tofile(f)
        return out_fp

    @staticmethod
    def _compute_mean(raw_data, fft_size, n_fft, fft_shift, refresh, dur_per_fft_us, data_dtypes):
        fft_cnt = 0
        hold_buffer = np.zeros(fft_size, dtype=np.complex64)
        hold_buffer_f = hold_buffer.view(np.float32)
        rbw = 1.0
        inv_rbv = 1 / rbw
        buf = []

        for i in range(n_fft):
            off = i * fft_size
            fft_buf = fft.fft(raw_data[off:off + fft_size], fft_size)

            # Accumulate the complex numbers
            hold_buffer += np.concatenate((fft_buf[fft_shift:fft_shift + fft_size - fft_shift],
                                           fft_buf[:fft_shift]))

            fft_cnt += 1
            if fft_cnt == refresh:
                # Compute the energy in dB performing the proper normalization
                # before any dB calculation, emulating the mean
                #
                # implement volk_32fc_s32f_x2_power_spectral_density_32f
                inv_nrm_factor = 1 / (fft_cnt * fft_size)
                iq = (hold_buffer_f * inv_nrm_factor).reshape((-1, 2))
                wr_buf = 10.0 * np.log10((np.add(*np.hsplit(iq * iq, 2)) + 1e-20) * inv_rbv)

                x = np.empty(1, dtype=data_dtypes)
                x['tabs'][0] = i * dur_per_fft_us
                x['spec'][0] = wr_buf.flatten()
                buf.append(x)

                fft_cnt = 0
                hold_buffer.fill(0.0)

        return np.concatenate(buf)

    @staticmethod
    def _compute_max_hold(raw_data, fft_size, n_fft, fft_shift, refresh, dur_per_fft_us, data_dtypes):
        fft_cnt = 0
        hold_buffer = np.zeros(fft_size, dtype=np.complex64)
        hold_buffer_f = hold_buffer.view(np.float32)
        buf = []

        for i in range(n_fft):
            off = i * fft_size
            fft_buf = fft.fft(raw_data[off:off + fft_size], fft_size)
            shift_buf = np.concatenate((fft_buf[fft_shift:fft_shift + fft_size - fft_shift],
                                        fft_buf[:fft_shift]))

            # Normalization factor
            shift_buf *= 1 / fft_size

            # Compute the mag^2
            iq = shift_buf.view(np.float32).reshape((-1, 2))
            tmp_buf = np.add(*np.hsplit(iq * iq, 2)).flatten()

            # Max hold
            np.fmax(hold_buffer_f[:fft_size], tmp_buf, out=hold_buffer_f[:fft_size])

            fft_cnt += 1
            if fft_cnt == refresh:
                # Compute the energy in dB
                wr_buf = 10.0 * np.log10(hold_buffer_f[:fft_size] + 1e-20)

                x = np.empty(1, dtype=data_dtypes)
                x['tabs'][0] = i * dur_per_fft_us
                x['spec'][0] = wr_buf[:fft_size]
                buf.append(x)

                fft_cnt = 0
                hold_buffer.fill(0.0)

        return np.concatenate(buf)

    @staticmethod
    def _compute_decimation(raw_data, fft_size, n_fft, fft_shift, refresh, dur_per_fft_us, data_dtypes):
        fft_cnt = 0
        hold_buffer = np.zeros(fft_size, dtype=np.complex64)
        hold_buffer_f = hold_buffer.view(np.float32)
        rbw = 1.0
        inv_rbv = 1 / rbw
        buf = []

        for i in range(n_fft):
            fft_cnt += 1
            if fft_cnt != refresh:
                continue

            off = i * fft_size
            fft_buf = fft.fft(raw_data[off:off + fft_size], fft_size)
            shift_buf = np.concatenate((fft_buf[fft_shift:fft_shift + fft_size - fft_shift],
                                        fft_buf[:fft_shift]))

            # Compute the energy in dB
            #
            # implement volk_32fc_s32f_x2_power_spectral_density_32f
            inv_nrm_factor = 1 / fft_size
            iq = (shift_buf * inv_nrm_factor).reshape((-1, 2))
            hold_buffer_f[:fft_size] = (10.0 * np.log10((np.add(*np.hsplit(iq * iq, 2)) + 1e-20) * inv_rbv)).view(np.float32).flatten()

            x = np.empty(1, dtype=data_dtypes)
            x['tabs'][0] = i * dur_per_fft_us
            x['spec'][0] = hold_buffer_f[:fft_size]
            buf.append(x)

            fft_cnt = 0

        return np.concatenate(buf)

    def plot(self, out_fn, vmin=None, vmax=None):
        tmin = self.start_timestamp + np.min(self.data['tabs'] / 1000000.0)
        tmax = self.start_timestamp + np.max(self.data['tabs'] / 1000000.0)
        fmin = np.min(self.freq / 1000.0)
        fmax = np.max(self.freq / 1000.0)
        if vmin is None or vmax is None:
            vmin = -100
            vmax = -50
            c_idx = self.data['spec'] > -200.0
            if np.sum(c_idx) > 100:
                data_mean = np.mean(self.data['spec'][c_idx])
                data_std = np.std(self.data['spec'][c_idx])
                vmin = data_mean - 2.0 * data_std
                vmax = data_mean + 6.0 * data_std

        plt.figure(figsize=(10, 20))
        plt.imshow(self.data['spec'],
                   origin='lower',
                   aspect='auto',
                   interpolation='None',
                   extent=[fmin, fmax, tmin, tmax],
                   vmin=vmin,
                   vmax=vmax,
                   cmap='viridis')

        plt.gca().yaxis.set_major_formatter(lambda x, pos=None: dt.datetime.utcfromtimestamp(x).strftime('%H:%M'))
        plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60))
        ax2 = plt.gca().secondary_yaxis('right', functions=(lambda x: x - self.start_timestamp,
                                                            lambda x: x + self.start_timestamp))
        ax2.set_ylabel('Time, s')

        plt.xlabel('Frequency, kHz')
        plt.ylabel('Time UTC')

        fig = plt.colorbar(aspect=50, pad=0.1)
        fig.set_label('Power, dB')

        plt.savefig(out_fn, bbox_inches='tight', dpi=200)
        plt.close()

    def spec_compress(self):
        spec = self.data['spec']
        std = np.std(spec, axis=0)
        off = np.mean(spec, axis=0) + self.OFFSET_IN_STDS * std
        scale = self.SCALE_IN_STDS * std / 255
        vals = np.clip((spec - off) / scale, 0.0, 255.0).astype(np.uint8)
        return off, scale, vals

    @staticmethod
    def spec_decompress(f, fft_size, n_fft):
        off = np.fromfile(f, np.float32, fft_size)
        scale = np.fromfile(f, np.float32, fft_size)
        vals = np.fromfile(f, np.uint8).reshape(n_fft, fft_size).astype(np.float32)
        return vals * scale + off


def numbi_disp(number, zero=None):
    """
    Actual for data sizes in bytes
    """

    try:
        number = len(number)
    except TypeError:
        pass

    if not number or number <= 0:
        if zero is not None:
            return zero
        number = 0

    # rememberings on BinkleyTerm
    rgch_size = 'bKMGTPEZY'
    i = 0
    oldq = 0
    quotient = number
    while quotient >= 1024:
        oldq = quotient
        quotient = oldq >> 10
        i += 1

    intq = quotient
    if intq > 999:
        # If more than 999 but less than 1024, it's a big fraction of
        # the next power of 1024. Get top two significant digits
        # (so 1023 would come out .99K, for example)
        intq = (intq * 25) >> 8   # 100/1024
        e_stuff = '.%2d%s' % (intq, rgch_size[i + 1])
    elif intq < 10 and i:
        # If less than 10 and not small units, then get some decimal
        # places (e.g. 1.2M)
        intq = (oldq * 5) >> 9    # 10/1024
        tempstr = '%02d' % intq
        e_stuff = '%s.%s%s' % (tempstr[0], tempstr[1], rgch_size[i])
    else:
        # Simple case. Just do it.
        e_stuff = '%d%s' % (intq, rgch_size[i])

    return e_stuff


def num_disp(num, res=2):
    mag = 0

    while abs(num) >= 1000:
        mag += 1
        num /= 1000

    return f"{num:.{res}f}".rstrip('0').rstrip('.') + ('', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')[mag]


def sec(t, res=2):
    return ('%.*f' % (res, t)).rstrip('0').rstrip('.') + 's'


def doppler_shift(freq: Union[int, float], vel: Union[int, float]):
    """
    Calculate Doppler shift by relative velocity

    :param freq: base signal frequency
    :param vel: relative velocity, m/s
    :return: Result frequency with doppler shift. NOTE: if vel is negative, result match for UPLINK, else for DOWNLINK
    """

    return freq * ephem.c / (ephem.c + vel)


def azimuth(a_lonlat: [float, float], b_lonlat: [float, float]) -> float:
    """
    Calculate azimuth between two points
    :param a_lonlat: Point A lonlat, radians
    :param b_lonlat: Point B lonlat, radians
    :return: azimuth in radians
    """

    lon_a, lat_a = a_lonlat
    lon_b, lat_b = b_lonlat

    if lon_b - lon_a < -math.pi:
        delta_lon = math.tau + lon_b - lon_a
    elif lon_b - lon_a > math.pi:
        delta_lon = lon_b - lon_a - math.tau
    else:   # abs(lon_b - lon_a) <= math.pi
        delta_lon = lon_b - lon_a

    return math.atan2(
        math.sin(delta_lon),
        math.cos(lat_a) * math.tan(lat_b) - math.sin(lat_a) * math.cos(delta_lon)
    )


def mktmp(dir: pathlib.Path = None, prefix: str = None, suffix='.tmp') -> pathlib.Path:
    if dir:
        dir.mkdir(parents=True, exist_ok=True)
    f = tempfile.NamedTemporaryFile(dir=dir, prefix=prefix, suffix=suffix, delete=False)
    f.close()
    return pathlib.Path(f.name)


def mktmp2(mode='w+b', buffering=-1, dir: pathlib.Path = None, prefix: str = None, suffix='.tmp'):
    if dir:
        dir.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(mode=mode,
                                       buffering=buffering,
                                       dir=dir,
                                       prefix=prefix,
                                       suffix=suffix,
                                       delete=False)


def close(*ff) -> None:
    for f in ff:
        try:
            if hasattr(f, 'close'):
                f.close()
            elif f is not None and f >= 0:
                os.close(f)
        except OSError:
            pass


def unlink(*pp: pathlib.Path) -> None:
    for p in pp:
        try:
            p.unlink(True)
        except OSError as e:
            if e.errno == errno.EISDIR:
                try:
                    p.rmdir()
                except:
                    pass


def img_add_exif(img: Image.Image,
                 d: dt.datetime = None,
                 observer: ephem.Observer = None,
                 comment='') -> Image.Image:
    exif = img.getexif()

    exif[ExifTags.Base.Software] = 'SatsReceiver'   # TODO: add version
    if d is not None:
        exif[ExifTags.Base.DateTime] = d.strftime('%Y:%m:%d %H:%M:%S')

    if observer is not None:
        img.info['exif'] = exif.tobytes()
        img = ImageOps.exif_transpose(img)
        exif = img.getexif()

        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps[ExifTags.GPS.GPSLatitudeRef] = 'S' if observer.lat < 0 else 'N'
        gps[ExifTags.GPS.GPSLatitude] = list(map(lambda x: abs(float(x)), str(observer.lat).split(':')))
        gps[ExifTags.GPS.GPSLongitudeRef] = 'W' if observer.lon < 0 else 'E'
        gps[ExifTags.GPS.GPSLongitude] = list(map(lambda x: abs(float(x)), str(observer.lon).split(':')))
        gps[ExifTags.GPS.GPSAltitudeRef] = int(observer.elev < 0)
        gps[ExifTags.GPS.GPSAltitude] = abs(observer.elev)
        exif[ExifTags.IFD.GPSInfo] = gps

    if comment:
        img.info['exif'] = exif.tobytes()
        img = ImageOps.exif_transpose(img)
        exif = img.getexif()

        ee = exif.get_ifd(ExifTags.IFD.Exif)
        ee[ExifTags.Base.UserComment] = comment
        exif[ExifTags.IFD.Exif] = ee

    img.info['exif'] = exif.tobytes()

    return ImageOps.exif_transpose(img)


_HPA_MMHG_CONST = 760 / 101325
def hpa_to_mmhg(hpa: float):
    return _HPA_MMHG_CONST * hpa * 100


def tle_calc_checksum(full_line: str):
    checksum = 0
    for c in full_line[:-1]:
        if c.isnumeric():
            checksum += int(c)
        elif c == '-':
            checksum += 1
    return str(checksum)[-1]


def tle_generate(name, l1, l2, ignore_checksum=0, log=None):
    for i in range(2):
        try:
            return ephem.readtle(name, l1, l2), (name, l1, l2)
        except ValueError as e:
            if str(e).startswith('incorrect TLE checksum'):
                cs1, cs2 = tle_calc_checksum(l1), tle_calc_checksum(l2)
                if log:
                    log.warning('%s: for `%s` expect %s:%s, got %s:%s%s',
                                e, name,
                                cs1, cs2, l1[-1], l2[-1],
                                ignore_checksum and '. Ignore' or '')
                if not ignore_checksum:
                    break
                l1 = l1[:-1] + cs1
                l2 = l2[:-1] + cs2
            else:
                raise e
