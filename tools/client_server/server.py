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

import gnuradio as gr
import gnuradio.gr
import numpy as np

from PIL import Image
from sats_receiver.async_signal import AsyncSignal
from sats_receiver.systems.apt import Apt
from sats_receiver.utils import Decode, MapShapes, numbi_disp, close, Waterfall, WfMode, RawFileType, unlink

from tools.client_server import gr_decoder
from tools.client_server import common as cli_srv_common


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

    def __init__(self, q, map_shapes, satdump, worker_task_file, worker_timeout=None):
        super().__init__()

        self.q = q
        self.map_shapes = map_shapes
        self.satdump = satdump
        self.worker_task_file = worker_task_file
        self.worker_timeout = worker_timeout

        self.lock = mp.Lock()
        self.rd, self.wr = mp.Pipe(False)

        try:
            worker_task_file.touch(exist_ok=False)
            worker_task_file.write_text('{}')
        except FileExistsError:
            pass

        for i in json.load(worker_task_file.open('r')):
            self.wr.send(pathlib.Path(i))

    def put(self, params, fp):
        if self.wr:
            with self.lock:
                data = json.load(self.worker_task_file.open('r'))
                data[str(fp)] = params
                json.dump(data, self.worker_task_file.open('w'))

            self.wr.send(fp)

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

            task = self.rd.recv()

            if task == '.':
                break

            try:
                with self.lock:
                    data = json.load(self.worker_task_file.open('r'))
                params = data[str(task)]
                dtype = Decode[params['decoder_type']]
                res = fp = task
                flush = 1
            except:
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
                try:
                    self._draw_wf(fp, params)
                except:
                    self.log.exception('waterfall')

            elif dtype == Decode.PROTO_RAW:
                try:
                    self._draw_wf(fp, params)
                except:
                    self.log.error('waterfall')

                try:
                    retry_cnt = 2
                    while retry_cnt:
                        self.log.info('Decode processing (to=%s)', self.worker_timeout)
                        if not gr_decoder.process2(fp, params, self.log, self.q, self.worker_timeout):
                            retry_cnt = 0
                            flush = 1
                        else:
                            self.log.warning('Terminate by timeout. Retry...')
                            retry_cnt -= 1
                            flush = 0
                except:
                    self.log.error('gr_decoder')

            if flush:
                with self.lock:
                    data = json.load(self.worker_task_file.open('r'))
                    data.pop(str(task))
                    json.dump(data, self.worker_task_file.open('w'))
            # else:
            #     self.wr.send(task)

            self.log.info('%s done: %s', params['sat_name'], res.name)

    def _draw_wf(self, fp, params):
        self.log.info('Draw Waterfall')

        wf = ofp = 0
        ftype = RawFileType[params['file_type']]
        if ftype == RawFileType.IQ:
            ofp = fp.with_stem(fp.stem + '_wav').with_suffix('.wfc')
            wf = Waterfall.from_wav(fp, ofp, end_timestamp=dt.datetime.fromisoformat(params['end_time']).timestamp())
        elif ftype == RawFileType.WFC:
            wf = Waterfall(fp)
        elif ftype != RawFileType.AUDIO:
            self.log.warning('Unknown filetype: %s', ftype)
        if wf:
            wf.plot(fp.with_stem(f'{fp.stem}_{ftype.name}_wf').with_suffix('.jpg'),
                    *params.get('wf_minmax', (None, None)))
        if ofp:
            unlink(ofp)


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
        try:
            self.prefix = '%s:%s' % client_address
        except:
            self.prefix = '%s' % client_address
        self.log = logging.getLogger('Handler: ' + self.prefix)
        self.ctx = None
        super().__init__(request, client_address, server)

    def send_reply(self, *a):
        if a:
            self.wfile.write(cli_srv_common.pack_reply(*a))

    def handle(self):
        try:
            self._handle()
        except KeyError as e:
            self.log.error('Missing key: %s', e)
        except:
            self.log.error('_handle', exc_info=True)
        finally:
            try:
                self.send_reply(cli_srv_common.REPLY_CMD_END)
            except:
                pass

    def _handle(self):
        hdr_ = self.rfile.read1(cli_srv_common.HDR.size)
        if not hdr_:
            return

        try:
            hdr = cli_srv_common.Hdr._make(cli_srv_common.HDR.unpack(hdr_))
        except (struct.error, TypeError) as e:
            self.log.error('%s: %s', e, hdr_)
            return

        e = cli_srv_common.verify_hdr(hdr)
        if e:
            cause, e = e
            self.log.error('Invalid %s: %s', e, hdr)
            self.send_reply(*e)
            return

        if hdr.cmd == cli_srv_common.HDR_CMD_SEND:
            e = self.cmd_send(hdr)
            if e:
                self.send_reply(*e)
                return

    def cmd_send(self, hdr: cli_srv_common.Hdr):
        data = self.rfile.read1(hdr.sz)
        try:
            params = json.loads(data.decode())
        except json.JSONDecodeError as e:
            self.log.error('Invalid JSON: %s', e)
            return cli_srv_common.REPLY_CMD_ERR,

        s = io.StringIO()
        s.write(str(self.client_address) + '\n')
        pprint.pprint(params, stream=s)
        s.seek(0)
        self.log.debug(s.read())

        clients = json.load(self.server.client_list.open())
        client_info = clients.get(params['secret'])
        if not client_info:
            self.log.debug('Unknown secret: %s', params['secret'])
            return cli_srv_common.REPLY_CMD_ERR,

        try:
            dtype = Decode[params['decoder_type']]
        except ValueError:
            self.log.warning('Invalid dtype `%s`. Skip', params['decoder_type'])
            return cli_srv_common.REPLY_CMD_ERR,

        self.log.info('%s for %s', dtype, params['sat_name'])

        out_dir = self.server.recv_path / params['sat_name']
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / params['filename']

        self.recv_file(fp, params['fsize'], params['compress'])

        self.server.worker.put(params, fp)

    def recv_file(self, fp: pathlib.Path, fsz: int, compress: bool):
        numfsz = numbi_disp(fsz)
        sz_left = fsz
        t = time.monotonic()
        t_next = t + 1

        mode = 'ab'
        if compress:
            # resume download not awailable
            zo = zlib.decompressobj(wbits=-9)
            mode = 'wb'
        with fp.open(mode) as f:
            off = f.tell()
            sz_left -= off
            self.send_reply(cli_srv_common.REPLY_CMD_RDY, off)

            while sz_left:
                data = self.request.recv(self.server.buf_sz)
                if not data:
                    if compress:
                        sz_left -= f.write(zo.flush(zlib.Z_FINISH))
                    if f.tell() == fsz:
                        break
                    raise ConnectionError('Connection lost')

                if not compress:
                    sz_left -= f.write(data)
                else:
                    sz_left -= f.write(zo.decompress(data))
                    sz_left -= f.write(zo.flush(zlib.Z_FULL_FLUSH))

                t = time.monotonic()
                if t > t_next:
                    t_next = t + 10
                    self.log.debug('%s/%s', numbi_disp(fsz - sz_left), numfsz)

            f.flush()

        self.log.debug('%s %s/%s', fp.name, numbi_disp(fsz - sz_left), numfsz)


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

    def get_request(self):
        conn, addr = super().get_request()
        return self.ssl_ctx.wrap_socket(conn, server_side=True), addr


class TcpServer(socketserver.ThreadingMixIn, _TcpServer):
    block_on_close = 0
    daemon_threads = 1


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

    gr_logger = gr.gr.logging()
    gr_logger.set_default_level(gr.gr.log_levels.err)


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
    worker_timeout = config.get('worker_timeout', None)
    worker_task_file = pathlib.Path(config['worker_task_file']).expanduser().absolute()

    q = mp.Queue()
    setup_logging(q, logging.DEBUG)
    logging.info('Hello!')

    worker = Worker(q, map_shapes, satdump, worker_task_file, worker_timeout)
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
