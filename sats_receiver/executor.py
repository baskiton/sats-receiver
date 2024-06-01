import logging
import logging.handlers
import multiprocessing as mp
import pathlib

from sats_receiver import utils


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
                self.stop()
            except:
                self.log.exception('stop Exception:')

        self.log.debug('finish')

    def action(self):
        while 1:
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
                if decoder_type == utils.Decode.RAW:
                    _, sat_name, observation_key, res_filename, end_time = x

                elif decoder_type == utils.Decode.CSOFT:
                    _, sat_name, observation_key, res_filename, end_time = x

                elif decoder_type == utils.Decode.CCSDSCC:
                    _, sat_name, observation_key, res_filename, end_time = x

                elif decoder_type == utils.Decode.APT:
                    _, sat_name, observation_key, res_filename, end_time = x

                elif decoder_type == utils.Decode.SSTV:
                    _, sat_name, observation_key, fn_dt = x

                elif decoder_type == utils.Decode.SATS:
                    _, sat_name, observation_key, files = x

    def execute(self, fn, *args, **kwargs):
        if self.wr:
            self.wr.send((fn, args, kwargs))

    def stop(self):
        if self.wr:
            self.wr.send('.')
            utils.close(self.wr)
            self.wr = 0
