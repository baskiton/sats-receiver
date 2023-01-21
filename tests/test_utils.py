import datetime as dt
import pathlib
import time
import threading
import queue

from typing import Mapping
from test import support
from unittest import TestCase

import numpy as np

from sats_receiver import utils


HERE = pathlib.Path(__file__).parent
TIMEOUT = support.SHORT_TIMEOUT


class SchedThread(threading.Thread):
    def __init__(self, sch: utils.Scheduler):
        super().__init__()
        self._q = queue.SimpleQueue()
        self.sch = sch

    def run(self):
        while 1:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            else:
                break

            d: dt.timedelta = self.sch.action()
            if d:
                time.sleep(d.seconds)

    def stop(self):
        self._q.put_nowait(1)


class TestScheduler(TestCase):
    """
    Test case examples are taken from the python repository
    https://github.com/python/cpython/blob/main/Lib/test/test_sched.py
    """

    def setUp(self):
        self._sch = utils.Scheduler()
        self._now = dt.datetime.now(dt.timezone.utc)
        self._l = []

    def test_plan(self):
        for x in [5, 4, 3, 2, 1]:
            self._sch.plan(self._now + dt.timedelta(seconds=x / 100), self._l.append, x, prior=1)
        time.sleep(0.2)
        self._sch.action()

        self.assertEqual([1, 2, 3, 4, 5], self._l)

    def test_plan_delay(self):
        self._sch.plan(self._now + dt.timedelta(seconds=1), self._l.append, 1, prior=1)
        delay = self._sch.action()

        self.assertGreaterEqual(dt.timedelta(seconds=1), delay)
        self.assertLessEqual(dt.timedelta(seconds=0), delay)

    def test_plan_concurrent(self):
        q = queue.SimpleQueue()
        fn = q.put
        self._sch.plan(self._now + dt.timedelta(seconds=0.1), fn, 1)
        self._sch.plan(self._now + dt.timedelta(seconds=0.3), fn, 3)
        t = SchedThread(self._sch)
        t.start()

        try:
            self.assertEqual(1, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            now = dt.datetime.now(dt.timezone.utc)
            for x in [4, 5, 2]:
                self._sch.plan(now + dt.timedelta(seconds=(x - 1) / 10), fn, x)

            self.assertEqual(2, q.get(timeout=TIMEOUT))
            self.assertEqual(3, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            self.assertEqual(4, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            self.assertEqual(5, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            time.sleep(1)
        finally:
            t.stop()
            t.join()

        self.assertTrue(q.empty())

    def test_priority(self):
        cases = [
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
            ([1, 2, 3, 4, 5], [2, 5, 3, 1, 4]),
            ([1, 1, 2, 2, 3], [1, 2, 3, 2, 1]),
        ]
        for expected, priorities in cases:
            with self.subTest(expected=expected, priorities=priorities):
                for priority in priorities:
                    self._sch.plan(self._now, self._l.append, priority, prior=priority)
                self._sch.action()
                self.assertEqual(expected, self._l)

                self.assertTrue(self._sch.empty())
                self._l.clear()

    def test_cancel(self):
        event1 = self._sch.plan(self._now + dt.timedelta(seconds=0.01), self._l.append, 1)
        event2 = self._sch.plan(self._now + dt.timedelta(seconds=0.02), self._l.append, 2)
        event3 = self._sch.plan(self._now + dt.timedelta(seconds=0.03), self._l.append, 3)
        event4 = self._sch.plan(self._now + dt.timedelta(seconds=0.04), self._l.append, 4)
        event5 = self._sch.plan(self._now + dt.timedelta(seconds=0.05), self._l.append, 5)
        self._sch.cancel(event1, event5)
        time.sleep(0.2)
        self._sch.action()

        self.assertEqual([2, 3, 4], self._l)

    def test_cancel_same_time(self):
        self._sch.plan(self._now, self._l.append, 'a')
        b = self._sch.plan(self._now, self._l.append, 'b')
        self._sch.plan(self._now, self._l.append, 'c')
        self._sch.cancel(b)
        self._sch.action()

        self.assertEqual(['a', 'c'], self._l)

    def test_cancel_expired_evt(self):
        event1 = self._sch.plan(self._now, self._l.append, 1)
        event2 = self._sch.plan(self._now, self._l.append, 2)
        self._sch.action()
        self._sch.cancel(event1, event2)

        self.assertTrue(self._sch.empty())
        self.assertEqual([1, 2], self._l)

    def test_cancel_invalid_evt(self):
        self._sch.plan(self._now, self._l.append, 1)
        self._sch.plan(self._now, self._l.append, 2)
        self._sch.cancel()
        self._sch.cancel(1)
        self._sch.cancel(2.0, '3', None)
        self.assertFalse(self._sch.empty())

    def test_cancel_concurrent(self):
        q = queue.SimpleQueue()
        fn = q.put
        event1 = self._sch.plan(self._now + dt.timedelta(seconds=0.1), fn, 1)
        event2 = self._sch.plan(self._now + dt.timedelta(seconds=0.2), fn, 2)
        event4 = self._sch.plan(self._now + dt.timedelta(seconds=0.4), fn, 4)
        event5 = self._sch.plan(self._now + dt.timedelta(seconds=0.5), fn, 5)
        event3 = self._sch.plan(self._now + dt.timedelta(seconds=0.3), fn, 3)
        t = SchedThread(self._sch)
        t.start()

        try:
            self.assertEqual(1, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            self._sch.cancel(event2, event5)
            self.assertTrue(q.empty())

            self.assertEqual(3, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            self.assertEqual(4, q.get(timeout=TIMEOUT))
            self.assertTrue(q.empty())

            time.sleep(1)
        finally:
            t.stop()
            t.join()

        self.assertTrue(q.empty())

    def test_empty(self):
        self.assertTrue(self._sch.empty())

        for x in [0.05, 0.04, 0.03, 0.02, 0.01]:
            self._sch.plan(self._now + dt.timedelta(seconds=x), self._l.append, x)

        self.assertFalse(self._sch.empty())

        time.sleep(0.2)
        self._sch.action()

        self.assertTrue(self._sch.empty())

    def test_clear(self):
        for x in [0.05, 0.04, 0.03, 0.02, 0.01]:
            self._sch.plan(self._now + dt.timedelta(seconds=x), self._l.append, x)
        self._sch.clear()

        self.assertTrue(self._sch.empty())

    def test_args_kwargs(self):
        def fn(*a, **b):
            self._l.append((a, b))

        self._sch.plan(self._now, fn)
        self._sch.plan(self._now, fn, 1, 2)
        self._sch.plan(self._now, fn, 'a', 'b')
        self._sch.plan(self._now, fn, 1, 2, foo=3)
        self._sch.action()
        self.assertCountEqual([
            ((), {}),
            ((1, 2), {}),
            (('a', 'b'), {}),
            ((1, 2), {'foo': 3})
        ], self._l)

    def test_plan_time(self):
        def fn():
            self._l.append(dt.datetime.now(dt.timezone.utc))

        t = self._now + dt.timedelta(seconds=1)
        self._sch.plan(t, fn)
        time.sleep(1)
        self._sch.action()

        self.assertLess(t, self._l[0])
        self.assertGreater(t + dt.timedelta(seconds=1), self._l[0])


class TestUtils(TestCase):
    def test_sysu_collect_debug_log(self):
        sysu = utils.SysUsage('Test', 0)
        with self.assertLogs(level='DEBUG'):
            sysu.collect()

    def test_numbi_disp_ok(self):
        cases = (
            ('0b', -1),
            ('0b', 0),
            ('5b', 'hello'),
            ('5b', b'world'),
            ('3b', [1, 2, 3]),
            ('1b', 1.9),
            ('999b', 999),
            ('.99K', 1023),
            ('1.0K', 1024),
            ('4.5K', (4 << 10) + 512),
            ('1.0M', 1 << 20),
            ('2.0G', 2 << 30),
            ('3.0T', 3 << 40),
            ('4.0P', 4 << 50),
            ('5.0E', 5 << 60),
            ('6.0Z', 6 << 70),
            ('7.0Y', 7 << 80),
            (12, -1, 12),
            ('-12', None, '-12'),
            (0, 0, 0),
        )
        for i in cases:
            with self.subTest(expect=i[0], num=i[1]):
                self.assertEqual(i[0], utils.numbi_disp(*i[1:]))

    def test_numbi_disp_type_error(self):
        with self.assertRaises(TypeError):
            utils.numbi_disp(self)

    def test_num_disp(self):
        cases = (
            ('-1', -1),
            ('0', 0),
            ('1.9', 1.9),
            ('999', 999),
            ('1k', 1005),
            ('1.02k', 1024),
            ('4.5k', 4 * 10e2 + 500),
            ('1M', 1 * 10e5),
            ('2G', 2 * 10e8),
            ('3T', 3 * 10e11),
            ('4P', 4 * 10e14),
            ('5E', 5 * 10e17),
            ('6Z', 6 * 10e20),
            ('7Y', 7 * 10e23),
            ('-1.2345k', -1234.5, 4),
            ('-1.2345k', -1234.5, 6),
        )
        for i in cases:
            with self.subTest(expect=i[0], num=i[1]):
                self.assertEqual(i[0], utils.num_disp(*i[1:]))

    def test_sec(self):
        cases = (
            ('855s', 855),
            ('12s', 12.3456, 0),
            ('12.3s', 12.3456, 1),
            ('12.35s', 12.3456, 2),
            ('12.346s', 12.3456, 3),
            ('12.3456s', 12.3456, 4),
            ('12.3456s', 12.3456, 6),
        )
        for i in cases:
            with self.subTest(expect=i[0], sec=i[1]):
                self.assertEqual(i[0], utils.sec(*i[1:]))

    def test_doppler_shift(self):
        cases = (
            (1000000, -1000),
            (1000000, -1),
            (1000000, 0),
            (1000000, 1),
            (1000000, 1000),
            (1, 10),
        )
        for freq, vel in cases:
            with self.subTest(freq=freq, vel=vel):
                if vel > 0:
                    a = self.assertGreater
                elif vel < 0:
                    a = self.assertLess
                else:
                    a = self.assertEqual

                a(freq, utils.doppler_shift(freq, vel))

    def test_azimuth(self):
        cases = (
            (np.radians(0), np.radians((0, 0)), np.radians((0, 90))),
            (np.radians(0), np.radians((0, -90)), np.radians((0, 90))),
            (np.radians(90), np.radians((0, 0)), np.radians((90, 0))),
            (np.radians(180), np.radians((0, 0)), np.radians((0, -90))),
            (np.radians(180), np.radians((0, 90)), np.radians((0, -90))),
            (np.radians(-90), np.radians((0, 0)), np.radians((-90, 0))),
            (np.radians(-90), np.radians((90, 0)), np.radians((0, 0))),
        )
        for expected, a_lonlat, b_lonlat in cases:
            with self.subTest(expected=expected, a_lonlat=a_lonlat, b_lonlat=b_lonlat):
                self.assertEqual(expected, utils.azimuth(a_lonlat, b_lonlat))

    def test_map_shapes_ok(self):
        color = 'coral', '#ff7f50', (0xff, 0x7f, 0x50), (0xff, 0x7f, 0x50, 255)

        cfg = dict(
            shapes_dir=HERE / 'files',
            line_width=1,
            shapes=(
                (1, 'ne_110m_graticules_30.zip', color[0]),
            ),
            points=dict(
                observer=dict(color=color[1], type='+', size=[3, 21]),
                another=dict(color=color[2], type='o', size=12, lonlat=[0, 0]),
                other=dict(color=color[3], type='+', size=[3, 21], lonlat=[81, -33]),
            ),
        )
        ms = utils.MapShapes(cfg)

        self.assertEqual(HERE / 'files', ms.shapes_dir)
        self.assertEqual(1, ms.line_width)

        for points, col in ms.iter():
            x = isinstance(points, Mapping)

            with self.subTest(points=points if x else 'shapefile', color=col):
                self.assertTupleEqual(color[3], col)

                if x:
                    if points['name'] != 'observer':
                        self.assertIn('lonlat', points)
                    if points['type'] == '+':
                        self.assertEqual(2, len(points['size']))
                    elif points['type'] == 'o':
                        self.assertIsInstance(points['size'], (int, float))

    def test_map_shapes_fail(self):
        cases = (
            (dict(points=dict(xxx=dict(type='+'))), KeyError),
            (dict(points=dict(xxx=dict(type='+', lonlat=[0, 0]))), KeyError),
            (dict(points=dict(xxx=dict(type='x', lonlat=[0, 0], color=None))), TypeError),
            (dict(points=dict(xxx=dict(type='x', lonlat=[0, 0], color='black'))), ValueError),
            (dict(points=dict(xxx=dict(type='+', lonlat=[0, 0], color=0))), KeyError),
            (dict(points=dict(xxx=dict(type='+', lonlat=[0, 0], color=0, size=None))), TypeError),
            (dict(points=dict(xxx=dict(type='+', lonlat=[0, 0], color=0, size=[None]))), AssertionError),
            (dict(points=dict(xxx=dict(type='o', lonlat=[0, 0], color=0, size=[None]))), TypeError),
        )
        for cfg, exc in cases:
            with self.subTest(cfg=cfg):
                with self.assertRaises(exc):
                    utils.MapShapes(cfg)
