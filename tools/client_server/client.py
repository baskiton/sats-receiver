import errno
import json
import logging
import logging.handlers
import os
import pathlib
import select
import ssl
import socket as sk
import threading
import time
import zlib

from collections import deque

from tools.client_server.server import SRV_HDR, SRV_HDR_SIGN, SRV_HDR_VER


class TcpSender(threading.Thread):
    def __init__(self, sendf, addr, cafile, secret, buffer_sz=8192, remove_success=False, compress=False):
        super().__init__()
        self.log = logging.getLogger('Sender')

        self.sendf = sendf
        self.addr = addr
        self.cafile = cafile
        self.secret = secret
        self.buffer_sz = buffer_sz
        self.remove_success = remove_success
        self.compress = compress

        try:
            self.sendf.touch(exist_ok=False)
            self.sendf.write_text('{\n}')
        except FileExistsError:
            pass

        self.lock = threading.Lock()
        self.data = json.load(self.sendf.open('r+'))
        self.q = deque(self.data)
        self._stop_rd, self._stop_wr = os.pipe()
        self.err_t_conn = 0
        self.err_dt_conn = 5

        self.ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cafile)
        self.ssl_ctx.check_hostname = 0

    def stop(self):
        os.write(self._stop_wr, b'.')

    def push(self, **data):
        with self.lock:
            key = pathlib.Path(data['filename']).name
            self.data.update({key: data})
            json.dump(self.data, self.sendf.open('w'))
            self.q.append(key)

    def send(self, data):
        fp = pathlib.Path(data['filename'])
        data['fsize'] = fp.stat().st_size
        data['filename'] = fp.name
        data['secret'] = self.secret
        data['compress'] = self.compress
        d_raw = json.dumps(data).encode()

        with self.ssl_ctx.wrap_socket(sk.create_connection(self.addr)) as ss:
            self.log.debug('send `%s`', fp.name)

            ss.send(SRV_HDR.pack(SRV_HDR_SIGN, SRV_HDR_VER, len(d_raw)))
            ss.send(d_raw)

            if self._send_routine(ss, fp):
                self.log.debug('send `%s` done', fp.name)
                return fp

    def _send_routine(self, ss: ssl.SSLSocket, fp: pathlib.Path):
        poller = select.poll()
        poller.register(ss, select.POLLIN)
        poller.register(self._stop_rd, select.POLLIN)
        to = None

        while 1:
            x = dict(poller.poll(to))
            if x:
                if self._stop_rd in x:
                    self.log.debug('send `%s` stop', fp.name)
                    return

                if ss.fileno() in x:
                    cmd = ss.recv(3)
                    if cmd == b'RDY':
                        to = 0
                        f = fp.open('rb')
                        if self.compress:
                            zo = zlib.compressobj(wbits=-9)

                    else:
                        return 1

            if to == 0:
                d = f.read(self.buffer_sz)
                if not d:
                    # no more data
                    if self.compress:
                        ss.send(zo.flush(zlib.Z_FINISH))
                    f.close()
                    return 1

                if not self.compress:
                    ss.send(d)
                else:
                    ss.send(zo.compress(d))
                    ss.send(zo.flush(zlib.Z_FULL_FLUSH))

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
                        fp = self.send(data.copy())
                        if not fp:
                            # terminate
                            break
                        self.err_dt_conn = 5
                        flush = 1
                        if self.remove_success:
                            fp.unlink(True)

                    except (ssl.SSLError, sk.gaierror, ConnectionError) as e:
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
                        json.dump(self.data, self.sendf.open('w'))
                    self.q.popleft()
