import atexit
import datetime as dt
import dateutil.tz
import json
import hashlib
import logging
import logging.handlers
import multiprocessing as mp
import pathlib

from sats_receiver import utils, HOMEDIR

from tools.client_server.client import TcpSender


SENDF = HOMEDIR / 'send.json'


class Executor(mp.Process):
    def __init__(self, q: mp.Queue = None, sysu_intv=utils.SysUsage.DEFAULT_INTV, config: pathlib.Path = None):
        super().__init__(daemon=True, name=self.__class__.__name__)

        self.q = q
        self.sysu_intv = sysu_intv
        self.rd, self.wr = mp.Pipe(False)

        config = json.load(config.expanduser().absolute().open())
        self.addr = config['host'], config.get('port', 443)
        self.cafile = pathlib.Path(config['cafile']).expanduser().absolute()
        with pathlib.Path(config['secretfile']).expanduser().absolute().open('rb') as fsec:
            self.secret = hashlib.sha256(fsec.read().strip()).hexdigest()
        self.buf_sz = config.get('buf_sz', 8192)
        self.remove_success = config.get('remove_success', False)

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

        self.sender = TcpSender(SENDF, self.addr, self.cafile, self.secret, self.buf_sz, self.remove_success)
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

                elif decoder_type == utils.Decode.RAW:
                    _, sat_name, observation_key, files, end_time = x
                    for ty, fp in files.items():
                        self.sender.push(decoder_type=dty,
                                         file_type=ty.value,
                                         sat_name=sat_name,
                                         observation_key=observation_key,
                                         filename=str(fp),
                                         end_time=str(end_time))

                elif decoder_type in (utils.Decode.CSOFT, utils.Decode.CCSDSCC, utils.Decode.APT):
                    _, sat_name, observation_key, res_filename, end_time = x
                    self.sender.push(decoder_type=dty,
                                     sat_name=sat_name,
                                     observation_key=observation_key,
                                     filename=str(res_filename),
                                     end_time=str(end_time))

                elif decoder_type == utils.Decode.SSTV:
                    _, sat_name, observation_key, fn_dt = x
                    for fn, end_time in fn_dt:
                        self.sender.push(decoder_type=dty,
                                         sat_name=sat_name,
                                         observation_key=observation_key,
                                         filename=str(fn),
                                         end_time=str(end_time))

                elif decoder_type == utils.Decode.SATS:
                    _, sat_name, observation_key, files = x
                    for ty_, files_ in files.items():
                        for fn in files_:
                            self.sender.push(decoder_type=dty,
                                             sat_name=sat_name,
                                             observation_key=observation_key,
                                             filename=str(fn),
                                             end_time=str(dt.datetime.fromtimestamp(
                                                 fn.stat().st_mtime, dateutil.tz.tzutc())))

                elif decoder_type == utils.Decode.PROTO:
                    _, deftype, sat_name, observation_key, res_filename, end_time = x
                    self.sender.push(decoder_type=dty,
                                     deframer_type=deftype.value,
                                     sat_name=sat_name,
                                     observation_key=observation_key,
                                     filename=str(res_filename),
                                     end_time=str(end_time))

    def execute(self, fn, *args, **kwargs):
        if self.wr:
            self.wr.send((fn, args, kwargs))

    def stop(self):
        if self.wr:
            self.wr.send('.')
            utils.close(self.wr)
            self.wr = 0
