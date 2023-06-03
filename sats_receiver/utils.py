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
import sys
import tempfile
import threading
import time

from typing import Any, Callable, Iterable, Mapping, Union

import ephem
import numpy as np
import psutil
import shapefile

from PIL import Image, ImageColor, ImageOps, ExifTags


THIRD_PI = math.pi / 3


class Mode(enum.Enum):
    RAW = 'RAW'
    AM = 'AM'
    FM = 'FM'
    WFM = 'WFM'
    WFM_STEREO = 'WFM_STEREO'
    QUAD = 'QUAD'
    QPSK = 'QPSK'
    GMSK = 'GMSK'


class Decode(enum.Enum):
    RAW = 'RAW'
    RSTREAM = 'RSTREAM'
    APT = 'APT'
    LRPT = 'LRPT'
    SSTV = 'SSTV'


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

    def plan(self, t: dt.datetime, fn: Callable, *a: Any, prior : int = 0, **kw: Any) -> Event:
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

    def action(self):
        pop = heapq.heappop
        while True:
            with self._lock:
                if not self._queue:
                    break
                t, prior, seq, fn, a, kw = self._queue[0]
                now = dt.datetime.now(dt.timezone.utc)

                if t > now:
                    delay = True
                else:
                    delay = False
                    pop(self._queue)

            if delay:
                return t - now

            fn(*a, **kw)
            # time.sleep(0)


class SysUsage:
    DEFAULT_INTV = 3600

    def __init__(self, ctx: str, intv : Union[int, float] = DEFAULT_INTV):
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
