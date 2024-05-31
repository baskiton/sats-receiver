import atexit
import datetime as dt
import dateutil.tz
import errno
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import pathlib
import select
import socket as sk
import threading
import time
import zlib

from collections import deque

from sats_receiver import utils, HOMEDIR


SENDF = HOMEDIR / 'send.json'
SERVER = 'deb', 7373


class Sender(threading.Thread):
    def __init__(self):
        super().__init__()
        self.log = logging.getLogger('Sender')

        try:
            SENDF.touch(exist_ok=False)
            SENDF.write_text('{\n}')
        except FileExistsError:
            pass

        self.lock = threading.Lock()
        self.data = json.load(SENDF.open('r+'))
        self.q = deque(self.data)
        self._stop_rd, self._stop_wr = os.pipe()
        self.err_t_conn = 0
        self.err_dt_conn = 5

    def stop(self):
        os.write(self._stop_wr, b'.')

    def push(self, **data):
        with self.lock:
            key = pathlib.Path(data['filename']).name
            self.data.update({key: data})
            json.dump(self.data, SENDF.open('w'))
            self.q.append(key)

    def send(self, data):
        fp = pathlib.Path(data['filename'])

        with sk.socket(sk.AF_INET, sk.SOCK_STREAM) as s:
            s.connect(SERVER)

            data['fsize'] = fp.stat().st_size
            data['filename'] = fp.name

            self.log.debug('send `%s`', fp.name)

            d_raw = json.dumps(data).encode()
            d_len = len(d_raw).to_bytes(4, 'little', signed=False)
            s.send(d_len)
            s.send(d_raw)

            while 1:
                ret = s.recv(3)
                if ret == b'RDY':
                    zo = zlib.compressobj(wbits=-9)
                    with fp.open('rb') as f:
                        d = f.read(8192)
                        while d:
                            s.send(zo.compress(d))
                            s.send(zo.flush(zlib.Z_FULL_FLUSH))
                            d = f.read(8192)
                        s.send(zo.flush(zlib.Z_FINISH))

                else:
                    break

        self.log.debug('send `%s` done', fp.name)

    def run(self):
        poller = select.poll()
        poller.register(self._stop_rd, select.POLLIN)

        while 1:
            if poller.poll(5000):
                break

            while self.q:
                flush = 0
                fn = self.q[0]
                data = self.data.get(fn)
                if not data:
                    self.log.warning('No data for %s. Skip', fn)

                else:
                    try:
                        self.send(data)
                        self.err_dt_conn = 5
                        flush = 1

                    except ConnectionError as e:
                        t = time.monotonic()
                        if t > self.err_t_conn:
                            self.err_t_conn = t + self.err_dt_conn
                            self.err_dt_conn *= 2
                            self.log.warning('send: %s', e.strerror)
                        break

                    except OSError as e:
                        if e.errno != errno.ENOENT:
                            t = time.monotonic()
                            if t > self.err_t_conn:
                                self.err_t_conn = t + self.err_dt_conn
                                self.err_dt_conn *= 2
                                self.log.warning('send: %s', e)
                            break
                        self.log.warning('send: %s', e)
                        flush = 1

                    except:
                        self.log.exception('send')

                with self.lock:
                    if flush:
                        self.data.pop(fn)
                        json.dump(self.data, SENDF.open('w'))
                    self.q.popleft()


class Executor(mp.Process):
    def __init__(self, q: mp.Queue = None, sysu_intv=utils.SysUsage.DEFAULT_INTV, config: pathlib.Path = None):
        super().__init__(daemon=True, name=self.__class__.__name__)

        self.q = q
        self.sysu_intv = sysu_intv
        self.rd, self.wr = mp.Pipe(False)

    def _setup_process(self):
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(mp.get_logger().level)
        if self.q is not None:
            qh = logging.handlers.QueueHandler(self.q)
            logger.addHandler(qh)
            # logging.basicConfig(level=mp.get_logger().level, handlers=[qh])
        self.log = logging.getLogger(self.name)
        self.sysu = utils.SysUsage(self.name, self.sysu_intv)

        self.sender = Sender()
        self.sender.start()
        atexit.register(lambda x: x.stop(), self.sender)

    def start(self) -> None:
        super(Executor, self).start()
        self.rd.close()

    def run(self):
        self._setup_process()

        self.log.debug('start')

        try:
            self.action()
        except:
            self.log.exception('Exception:')
        finally:
            try:
                self.sender.stop()
                self.stop()
            except:
                self.log.exception('stop Exception:')

        self.log.debug('finish')

    def action(self):
        while self.sender.is_alive():
            self.sysu.collect()

            try:
                x = self.rd.poll(1)
            except InterruptedError:
                x = 1
            except:
                continue

            if not x:
                continue

            x = self.rd.recv()

            if x == '.':
                break

            try:
                fn, args, kwargs = x
            except ValueError:
                self.log.error('invalid task: %s', x)
                continue

            if callable(fn):
                try:
                    x = fn(*args, **kwargs)
                except Exception:
                    self.log.exception('%s with args=%s kwargs=%s', fn, args, kwargs)
                    continue

                if not x:
                    continue

                decoder_type = x[0]
                dty = decoder_type.value

                if decoder_type == utils.Decode.NONE:
                    continue

                elif decoder_type in (utils.Decode.RAW, utils.Decode.CSOFT, utils.Decode.CCSDSCC, utils.Decode.APT):
                    _, sat_name, fin_key, res_filename, end_time = x
                    self.sender.push(decoder_type=dty,
                                     sat_name=sat_name,
                                     fin_key=fin_key,
                                     filename=str(res_filename),
                                     end_time=str(end_time))

                elif decoder_type == utils.Decode.SSTV:
                    _, sat_name, fin_key, fn_dt = x
                    for fn, end_time in fn_dt:
                        self.sender.push(decoder_type=dty,
                                         sat_name=sat_name,
                                         fin_key=fin_key,
                                         filename=str(fn),
                                         end_time=str(end_time))

                elif decoder_type == utils.Decode.SATS:
                    _, sat_name, fin_key, files = x
                    for ty_, files_ in files.items():
                        for fn in files_:
                            self.sender.push(decoder_type=dty,
                                             sat_name=sat_name,
                                             fin_key=fin_key,
                                             filename=str(fn),
                                             end_time=str(dt.datetime.fromtimestamp(
                                                 fn.stat().st_mtime, dateutil.tz.tzutc())))

    def execute(self, fn, *args, **kwargs):
        if self.wr:
            self.wr.send((fn, args, kwargs))

    def stop(self):
        if self.wr:
            self.wr.send('.')
            utils.close(self.wr)
            self.wr = 0


if __name__ == '__main__':
    from sats_receiver.async_signal import AsyncSignal

    def setup_logging(q: mp.Queue, log_lvl: int):
        if not isinstance(log_lvl, int):
            raise ValueError('Invalid log level: %s' % log_lvl)

        logger = logging.getLogger()
        logger.setLevel(log_lvl)
        logger.addHandler(logging.handlers.QueueHandler(q))
        mp.get_logger().setLevel(log_lvl)

        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(name)s: %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        qhl = logging.handlers.QueueListener(q, sh)
        qhl.start()
        atexit.register(qhl.stop)

    q = mp.Queue()
    setup_logging(q, logging.DEBUG)

    logging.info('Hello!')

    asig = AsyncSignal(['SIGABRT', 'SIGHUP', 'SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2', 'SIGBREAK'])
    ee = Executor(q)
    ee.start()
    atexit.register(lambda x: (x.stop(), x.join()), ee)

    while ee.is_alive():
        signame = asig.wait(1)
        if signame:
            if 'USR' in signame:
                # TODO
                pass
            else:
                logging.info('Exit by %s', signame)
                break

    logging.info('Bye!')
