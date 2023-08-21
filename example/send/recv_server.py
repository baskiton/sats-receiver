import atexit
import io
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import pathlib
import pprint
import sys
import socketserver as ss
import subprocess as sp
import threading
import time

import numpy as np

from PIL import Image
from sats_receiver.async_signal import AsyncSignal
from sats_receiver.systems.apt import Apt
from sats_receiver.utils import Decode, MapShapes, numbi_disp, close


RECV_PATH = pathlib.Path('/media/MORE/sats_receiver/records')
MAP_SHAPES = pathlib.Path('~/PycharmProjects/sats-receiver/map_shapes.json').expanduser().absolute()
SATDUMP = pathlib.Path('~/CLionProjects/satdump/build/satdump').expanduser().absolute()
SD_METEOR_72 = 'meteor_m2-x_lrpt'
SD_METEOR_80 = 'meteor_m2-x_lrpt_80k'


class Worker(mp.Process):
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

    def __init__(self, q):
        super().__init__()

        self.q = q
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
                sp.run([SATDUMP, SD_METEOR_80 if fp.stem.endswith('80k') else SD_METEOR_72, 'soft', str(fp), res],
                       cwd=SATDUMP.parent)

            if dtype == Decode.CCSDSCC:
                res = fp.parent / fp.stem
                res.mkdir(parents=True, exist_ok=True)
                fp = fp.rename(res / fp.name)
                sp.run([SATDUMP, SD_METEOR_80 if fp.stem.endswith('80k') else SD_METEOR_72, 'cadu', str(fp), res],
                       cwd=SATDUMP.parent)

            elif dtype == Decode.APT:
                res = self.apt_to_png_map(fp, MAP_SHAPES)

            self.log.info('%s done: %s', params['sat_name'], res.name)


class MyTCPHandler(ss.StreamRequestHandler):
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
        with fp.open('wb') as f:
            while sz_left:
                data = self.request.recv(4096)
                if not data:
                    raise ConnectionError('Connection lost')
                sz_left -= len(data)
                f.write(data)

                t = time.monotonic()
                if t > t_next:
                    t_next = t + 1
                    self.log.debug('%s %s/%s', fp.name, numbi_disp(fsz - sz_left), numfsz)

        self.log.debug('%s %s/%s', fp.name, numbi_disp(fsz - sz_left), numfsz)

    def handle(self):
        try:
            self._handle()
        except Exception:
            self.log.error('_handle', exc_info=True)
        finally:
            self.request.send(b'END')

    def _handle(self):
        l = int.from_bytes(self.request.recv(4), 'little', signed=False)
        params = json.loads(self.request.recv(l).decode())

        out_dir = RECV_PATH / params['sat_name']
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            dtype = Decode(params['decoder_type'])
        except ValueError:
            self.log.warning('Invalid dtype. Skip')
            return

        self.log.info('Receiving %s for %s', dtype, params['sat_name'])

        s = io.StringIO()
        s.write('\n')
        pprint.pprint(params, stream=s)
        s.seek(0)
        self.log.debug(s.read())

        self.request.send(b'RDY')

        fp = out_dir / params['filename']
        self.recv_file(fp, params['fsize'])

        self.server.worker.put(params, fp, dtype)


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
    q = mp.Queue()
    setup_logging(q, logging.DEBUG)
    logging.info('Hello!')

    asig = AsyncSignal(['SIGABRT', 'SIGHUP', 'SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2', 'SIGBREAK'])
    srv = ss.TCPServer(('0.0.0.0', 7373), MyTCPHandler)
    srv.worker = Worker(q)
    srv.worker.start()
    atexit.register(lambda x: (x.stop(), x.join()), srv.worker)

    srv_thr = threading.Thread(target=srv.serve_forever)
    srv_thr.start()
    atexit.register(lambda x: (srv.shutdown(), x.join()), srv_thr)

    while srv_thr.is_alive() and srv.worker.is_alive():
        signame = asig.wait(1)
        if signame:
            if 'USR' in signame:
                # TODO
                pass
            else:
                srv.shutdown()
                srv.worker.stop()
                logging.info('Exit by %s', signame)
                break

    logging.info('Bye!')
