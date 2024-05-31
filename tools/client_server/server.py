import argparse
import atexit
import datetime as dt
import io
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import pathlib
import pprint
import struct
import sys
import ssl
import socketserver
import subprocess as sp
import threading
import time
import zlib

import numpy as np

from PIL import Image
from sats_receiver.async_signal import AsyncSignal
from sats_receiver.systems.apt import Apt
from sats_receiver.utils import Decode, MapShapes, numbi_disp, close, Waterfall, WfMode


SRV_HDR = struct.Struct('!4sBI')    # SREC, ver, sz
SRV_HDR_SIGN = b'SREC'
SRV_HDR_VER = 1


class Worker(mp.Process):
    SD_METEOR_72 = 'meteor_m2-x_lrpt'
    SD_METEOR_80 = 'meteor_m2-x_lrpt_80k'

    @staticmethod
    def apt_to_png_map(apt_filename, config):
        if isinstance(config, (str, pathlib.Path)):
            config = json.load(open(config))

        apt_filename = pathlib.Path(apt_filename)
        result_dir = apt_filename.parent / apt_filename.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        # ret_fn = result_dir / (apt_filename.stem + '.png')
        apt_filename = apt_filename.rename(result_dir / apt_filename.name)
        ret_fn = apt_filename.with_suffix('.png')

        apt = Apt.from_apt(apt_filename)

        if apt.process():
            sys.exit(1)

        img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L')
        img.save(ret_fn, 'png')
        os.utime(ret_fn, (apt.end_time.timestamp(), apt.end_time.timestamp()))

        msh = MapShapes(config)
        apt.create_maps_overlay(msh)

        map_overlay = apt.map_overlay
        map_overlay = Image.fromarray(map_overlay, 'RGBA')
        map_overlay.save(ret_fn.with_stem(ret_fn.stem + '_map_overlay'), 'png')

        img = img.convert('RGB')
        img.paste(map_overlay, (apt.IMAGE_A_START, 0), map_overlay)
        img.paste(map_overlay, (apt.IMAGE_B_START, 0), map_overlay)
        img.save(ret_fn.with_stem(ret_fn.stem + '_map'), 'png')

        # comps = 'HVC', 'HVCT', 'MSA', 'NO', 'HVCP', 'HVCTP', 'MSAP', 'SEA', 'THRM', 'BP'
        comps = 'HVC', 'HVCT', 'MSA', 'NO', 'HVCTP', 'SEA', 'THRM', 'BP'
        for c, img in apt.create_composites(*comps):
            img.save(ret_fn.with_stem(f'{ret_fn.stem}_{c}'), 'png')

        return ret_fn

    def __init__(self, q, map_shapes, satdump):
        super().__init__()

        self.q = q
        self.map_shapes = map_shapes
        self.satdump = satdump
        self.rd, self.wr = mp.Pipe(False)

    def put(self, params, fp, dtype):
        if self.wr:
            self.wr.send((params, fp, dtype))

    def stop(self):
        if self.wr:
            self.wr.send('.')
            close(self.wr)
            self.wr = 0

    def _setup_process(self):
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(mp.get_logger().level)
        if self.q is not None:
            qh = logging.handlers.QueueHandler(self.q)
            logger.addHandler(qh)
        self.log = logging.getLogger(self.name)

    def start(self) -> None:
        super(Worker, self).start()
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
                self.stop()
            except:
                self.log.exception('stop Exception:')

        self.log.debug('finish')

    def action(self):
        while 1:
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
                params, fp, dtype = x
                res = fp
            except ValueError:
                self.log.error('invalid task: %s', x)
                continue

            self.log.info('%s process...', params['sat_name'])

            if dtype == Decode.CSOFT:
                res = fp.parent / fp.stem
                res.mkdir(parents=True, exist_ok=True)
                fp = fp.rename(res / fp.name)
                sp.run([self.satdump, self.SD_METEOR_80 if fp.stem.endswith('80k') else self.SD_METEOR_72, 'soft', str(fp), res],
                       cwd=self.satdump.parent)

            if dtype == Decode.CCSDSCC:
                res = fp.parent / fp.stem
                res.mkdir(parents=True, exist_ok=True)
                fp = fp.rename(res / fp.name)
                sp.run([self.satdump, self.SD_METEOR_80 if fp.stem.endswith('80k') else self.SD_METEOR_72, 'cadu', str(fp), res],
                       cwd=self.satdump.parent)

            elif dtype == Decode.APT:
                res = self.apt_to_png_map(fp, self.map_shapes)

            elif dtype == Decode.RAW:
                self.log.info('Draw Waterfall')
                try:
                    wf = Waterfall(fp, end_timestamp=dt.datetime.fromisoformat(params['end_time']).timestamp())
                    wf.plot(fp.with_stem(fp.stem + '_wf').with_suffix('.png'))
                except:
                    self.log.exception('waterfall')

            self.log.info('%s done: %s', params['sat_name'], res.name)


class MyTCPHandler(socketserver.StreamRequestHandler):
    @staticmethod
    def apt_to_png_map(apt_filename, config):
        if isinstance(config, (str, pathlib.Path)):
            config = json.load(open(config))

        apt_filename = pathlib.Path(apt_filename)
        result_dir = apt_filename.parent / apt_filename.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        # ret_fn = result_dir / (apt_filename.stem + '.png')
        apt_filename = apt_filename.rename(result_dir / apt_filename.name)
        ret_fn = apt_filename.with_suffix('.png')

        apt = Apt.from_apt(apt_filename)

        if apt.process():
            sys.exit(1)

        img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L')
        img.save(ret_fn, 'png')
        os.utime(ret_fn, (apt.end_time.timestamp(), apt.end_time.timestamp()))

        msh = MapShapes(config)
        apt.create_maps_overlay(msh)

        map_overlay = apt.map_overlay
        map_overlay = Image.fromarray(map_overlay, 'RGBA')

        img = img.convert('RGB')
        img.paste(map_overlay, (apt.IMAGE_A_START, 0), map_overlay)
        img.paste(map_overlay, (apt.IMAGE_B_START, 0), map_overlay)
        img.save(ret_fn.with_stem(ret_fn.stem + '_map'), 'png')

        # comps = 'HVC', 'HVCT', 'MSA', 'NO', 'HVCP', 'HVCTP', 'MSAP', 'SEA', 'THRM', 'BP'
        comps = 'HVC', 'HVCT', 'MSA', 'NO', 'HVCTP', 'SEA', 'THRM', 'BP'
        for c, img in apt.create_composites(*comps):
            img.save(ret_fn.with_stem(f'{ret_fn.stem}_{c}'), 'png')

        return ret_fn

    def __init__(self, request, client_address, server):
        self.log = logging.getLogger('Handler')
        super().__init__(request, client_address, server)

    def recv_file(self, fp: pathlib.Path, fsz):
        numfsz = numbi_disp(fsz)
        sz_left = fsz
        t = time.monotonic()
        t_next = t + 1
        zo = zlib.decompressobj(wbits=-9)
        with fp.open('wb') as f:
            while sz_left:
                data = self.rfile.read(self.server.buf_sz)
                if not data:
                    raise ConnectionError('Connection lost')
                # sz_left -= len(data)
                sz_left -= f.write(zo.decompress(data))
                sz_left -= f.write(zo.flush(zlib.Z_FULL_FLUSH))

                t = time.monotonic()
                if t > t_next:
                    t_next = t + 10
                    self.log.debug('%s %s/%s', fp.name, numbi_disp(fsz - sz_left), numfsz)

            f.write(zo.flush(zlib.Z_FINISH))

        self.log.debug('%s %s/%s', fp.name, numbi_disp(fsz - sz_left), numfsz)

    def handle(self):
        try:
            self._handle()
        except Exception:
            self.log.error('_handle from %s', self.client_address, exc_info=True)
        finally:
            self.wfile.write(b'END')

    def _handle(self):
        hdr_b = self.rfile.read(SRV_HDR.size)
        sign, ver, sz = hdr = SRV_HDR.unpack_from(hdr_b)
        if not (sign == SRV_HDR_SIGN and ver == SRV_HDR_VER):
            self.log.debug('%s -> Invalid HDR: %s (%s)', self.client_address, hdr, hdr_b)
            return

        data = self.rfile.read(sz)
        try:
            params = json.loads(data.decode())
        except:
            self.log.error('%s -> %s %s', self.client_address, hdr, data)
            return

        clients = set(json.load(self.server.client_list.open()))
        if params['secret'] not in clients:
            self.log.debug('%s -> Unknown secret: %s', self.client_address, params['secret'])
            return

        s = io.StringIO()
        s.write('\n')
        pprint.pprint(params, stream=s)
        s.seek(0)
        self.log.debug(s.read())

        try:
            dtype = Decode(params['decoder_type'])
        except ValueError:
            self.log.warning('%s -> Invalid dtype. Skip', self.client_address)
            return

        self.log.info('%s -> %s for %s', self.client_address, dtype, params['sat_name'])

        out_dir = self.server.recv_path / params['sat_name']
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / params['filename']

        self.wfile.write(b'RDY')
        self.recv_file(fp, params['fsize'])

        self.server.worker.put(params, fp, dtype)


class _TcpServer(socketserver.TCPServer):
    allow_reuse_address = 1
    allow_reuse_port = 1

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 client_list,
                 buf_sz,
                 recv_path,
                 certfile,
                 keyfile,
                 worker,
                 bind_and_activate=True):
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

        json.load(client_list.open())
        self.client_list = client_list
        self.buf_sz = buf_sz
        self.recv_path = recv_path
        self.ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ssl_ctx.load_cert_chain(certfile, keyfile)
        self.worker = worker
        self.worker.start()
        atexit.register(lambda x: (x.stop(), x.join()), self.worker)

    def get_request(self):
        conn, addr = super().get_request()
        return self.ssl_ctx.wrap_socket(conn, server_side=True), addr


class TcpServer(socketserver.ThreadingMixIn, _TcpServer):
    pass


def setup_logging(q, log_lvl):
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

    # PIL logging level
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.DEBUG + 2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=pathlib.Path, help='Config file path')
    args = ap.parse_args()

    config = json.load(args.config.expanduser().absolute().open())
    addr = config.get('host', ''), config.get('port', 443)
    client_list = pathlib.Path(config['client_list']).expanduser().absolute()
    recv_path = pathlib.Path(config['recv_path']).expanduser().absolute()
    cafile = pathlib.Path(config['cafile']).expanduser().absolute()
    keyfile = pathlib.Path(config['keyfile']).expanduser().absolute()
    map_shapes = pathlib.Path(config['map_shapes']).expanduser().absolute()
    satdump = pathlib.Path(config['satdump']).expanduser().absolute()
    buf_sz = config.get('buf_sz', 8192)

    q = mp.Queue()
    setup_logging(q, logging.DEBUG)
    logging.info('Hello!')

    worker = Worker(q, map_shapes, satdump)
    worker.start()
    atexit.register(lambda x: (x.stop(), x.join()), worker)

    asig = AsyncSignal(['SIGABRT', 'SIGHUP', 'SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2', 'SIGBREAK'])
    srv = TcpServer(addr, MyTCPHandler, client_list, buf_sz, recv_path, cafile, keyfile, worker)

    srv_thr = threading.Thread(target=srv.serve_forever)
    srv_thr.start()
    atexit.register(lambda x: (srv.shutdown(), x.join()), srv_thr)

    while srv_thr.is_alive() and worker.is_alive():
        signame = asig.wait(1)
        if signame:
            if 'USR' in signame:
                # TODO
                pass
            else:
                srv.shutdown()
                worker.stop()
                logging.info('Exit by %s', signame)
                break

    logging.info('Bye!')
