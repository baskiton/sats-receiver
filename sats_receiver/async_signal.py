import multiprocessing as mp
import os
import signal


SIG = {}
for n, v in signal.__dict__.items():
    if n.startswith('SIG') and n.isalnum() and isinstance(v, int):
        SIG[n] = v
        SIG[n[3:]] = v
        SIG[v] = n


class AsyncSignal:
    def __init__(self, signals=None):
        super(AsyncSignal, self).__init__()

        self.olds = {}

        if signals is None:
            signals = ['SIGINT']

        for s in signals:
            try:
                sig = int(s)
            except ValueError:
                sig = SIG.get(s, -1)

            if not (0 < sig < 256 and sig in SIG):
                continue

            old = signal.signal(sig, self._handler)
            if old is None:
                old = signal.SIG_DFL
            self.olds[sig] = old

        if self.olds:
            self.pid = os.getpid()
            self.rd, self.wr = mp.Pipe(False)

    def close(self):
        if self.olds:
            for sig, old in self.olds.items():
                signal.signal(sig, old)
            self.rd.close()
            self.wr.close()
        self.olds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _handler(self, signum, frame):
        if os.getpid() != self.pid:
            os.kill(self.pid, signum)
        else:
            self.wr.send(signum)

    def wait(self, t=None):
        try:
            x = self.rd.poll(t)
        except InterruptedError:
            x = 1

        if x:
            try:
                return SIG[self.rd.recv()]
            except:
                return
