import collections
import datetime as dt
import enum
import ephem
import gc
import heapq
import itertools
import logging
import math
import resource
import sched
import sys
import threading
import time


class Mode(enum.Enum):
    RAW = 'RAW'
    AM = 'AM'
    FM = 'FM'
    WFM = 'WFM'
    WFM_STEREO = 'WFM_STEREO'
    QUAD = 'QUAD'


class Decode(enum.Enum):
    RAW = 'RAW'
    APT = 'APT'
    LRPT = 'LRPT'


Event = collections.namedtuple('Event', 't, prior, seq, fn, a, kw')


class Scheduler:
    def __init__(self):
        self._queue = []
        self._lock = threading.RLock()
        self._sequence_generator = itertools.count()

    def plan(self, t: dt.datetime, fn, *a, prior=0, **kw):
        with self._lock:
            event = Event(t, prior, next(self._sequence_generator), fn, a, kw)
            heapq.heappush(self._queue, event)
        return event

    def cancel(self, *evt: sched.Event):
        for e in evt:
            try:
                with self._lock:
                    self._queue.remove(e)
                    heapq.heapify(self._queue)
            except ValueError:
                pass

    def clear(self):
        with self._lock:
            self._queue.clear()
            heapq.heapify(self._queue)

    def empty(self):
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
            else:
                fn(*a, **kw)
                time.sleep(0)


class MemMan:
    def __init__(self, intv=600):
        super(MemMan, self).__init__()
        gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
        self.now = 0
        self.intv = intv
        self.next = self.t + intv

    def collect(self):
        if self.t >= self.next:
            self.next = self.now + self.intv
            gc.collect()
            ru = resource.getrusage(resource.RUSAGE_SELF)
            logging.debug('MemMan: %s rss %s utime %s stime %s',
                          numdisp(sum(sys.getsizeof(i) for i in gc.get_objects())),
                          numdisp(ru.ru_maxrss << 10),
                          sec(ru.ru_utime),
                          sec(ru.ru_stime))

    @property
    def t(self):
        self.now = time.monotonic()
        return self.now


def numdisp(number, zero=None):
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
        e_stuff = ".%2d%s" % (intq, rgch_size[i + 1])
    elif intq < 10 and i:
        # If less than 10 and not small units, then get some decimal
        # places (e.g. 1.2M)
        intq = (oldq * 5) >> 9    # 10/1024
        tempstr = "%02d" % intq
        e_stuff = "%s.%s%s" % (tempstr[0], tempstr[1], rgch_size[i])
    else:
        # Simple case. Just do it.
        e_stuff = "%d%s" % (intq, rgch_size[i])

    return e_stuff


def sec(t, res=2):
    return ('%.*f' % (res, t)).rstrip('0').rstrip('.') + 's'


def doppler_shift(freq, vel):
    return freq * ephem.c / (ephem.c + vel)


def doppler_shift_rel(freq, vel):
    a = math.sqrt(1 - vel ** 2 / ephem.c ** 2)
    b = 1 - (vel / ephem.c) * math.cos(ephem.pi if vel < 0 else 0.0)
    return freq * (a / b)
